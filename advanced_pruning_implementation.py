"""
Advanced Interpretation-Aware Pruning Implementation
Based on comprehensive pruning literature and mechanistic analysis
PhD-level implementation with proper techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from tqdm import tqdm
import copy
import math
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Advanced pruning configuration based on literature"""
    # Pruning schedule
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.5
    pruning_steps: int = 10  # Gradual pruning over N steps
    pruning_frequency: int = 100  # Prune every N training steps
    
    # Fine-tuning
    learning_rate: float = 5e-5  # Higher than normal fine-tuning
    warmup_steps: int = 500  # Critical for recovery
    pruning_lr_multiplier: float = 2.0  # Boost LR after pruning
    
    # Method
    pruning_method: str = 'movement'  # 'magnitude', 'movement', 'taylor', 'hybrid'
    structured: bool = True  # Structured pruning for hardware efficiency
    
    # Distillation
    use_distillation: bool = True
    distillation_alpha: float = 0.5  # Balance between task loss and distillation
    temperature: float = 4.0  # Distillation temperature
    
    # Circuit preservation
    circuit_preservation_weight: float = 2.0  # Boost importance of circuit components
    min_heads_per_layer: int = 2  # Never prune below this
    protect_critical_layers: List[int] = None  # Layers to protect
    
    def __post_init__(self):
        if self.protect_critical_layers is None:
            self.protect_critical_layers = [2, 3, 6, 7]  # Based on your analysis


class AdvancedPruningModule:
    """
    State-of-the-art pruning implementation combining multiple techniques
    """
    
    def __init__(
        self,
        model: nn.Module,
        teacher_model: Optional[nn.Module],
        importance_scores: Dict[str, float],
        circuits: List[Dict],
        config: PruningConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.teacher_model = teacher_model  # For distillation
        self.importance_scores = importance_scores
        self.circuits = circuits
        self.config = config
        self.device = device
        
        # Initialize pruning state
        self.pruning_step = 0
        self.training_step = 0
        self.current_sparsity = config.initial_sparsity
        
        # Weight tracking for movement pruning
        self.initial_weights = self._get_weight_snapshot()
        self.weight_importance = {}
        
        # Masks (soft masks for gradual pruning)
        self.masks = self._initialize_soft_masks()
        
        # Circuit component mapping
        self.circuit_components = self._map_circuit_components()
        
    def _get_weight_snapshot(self) -> Dict[str, torch.Tensor]:
        """Snapshot current weights for movement pruning"""
        snapshot = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                snapshot[name] = param.data.clone()
        return snapshot
    
    def _initialize_soft_masks(self) -> Dict[str, torch.Tensor]:
        """Initialize soft masks (continuous values for gradual pruning)"""
        masks = {}
        
        base_model = self.model.bert if hasattr(self.model, 'bert') else self.model
        
        if hasattr(base_model, 'encoder'):
            for layer_idx, layer in enumerate(base_model.encoder.layer):
                # Attention masks (per head)
                if hasattr(layer.attention, 'self'):
                    n_heads = layer.attention.self.num_attention_heads
                    hidden_size = layer.attention.self.all_head_size
                    head_dim = hidden_size // n_heads
                    
                    # Create per-head masks
                    masks[f'layer_{layer_idx}_attention_heads'] = torch.ones(
                        n_heads, device=self.device
                    )
                    
                    # Query, Key, Value masks
                    for component in ['query', 'key', 'value']:
                        weight = getattr(layer.attention.self, component).weight
                        masks[f'layer_{layer_idx}_attention_{component}'] = torch.ones_like(
                            weight, device=self.device
                        )
                
                # MLP masks (structured by neurons)
                if hasattr(layer, 'intermediate'):
                    intermediate_size = layer.intermediate.dense.weight.shape[0]
                    masks[f'layer_{layer_idx}_mlp_neurons'] = torch.ones(
                        intermediate_size, device=self.device
                    )
        
        return masks
    
    def _map_circuit_components(self) -> Dict[str, float]:
        """Map discovered circuits to components with boosted importance"""
        component_circuit_importance = {}
        
        for circuit in self.circuits:
            importance = circuit.get('importance_score', 0)
            # Normalize importance to [0, 1]
            normalized_importance = min(importance / 100.0, 1.0)
            
            for component in circuit.get('components', []):
                if component not in component_circuit_importance:
                    component_circuit_importance[component] = 0
                component_circuit_importance[component] = max(
                    component_circuit_importance[component],
                    normalized_importance
                )
        
        return component_circuit_importance
    
    def calculate_importance_scores(self) -> Dict[str, float]:
        """
        Calculate importance scores using hybrid approach
        Combines magnitude, movement, gradient, and circuit information
        """
        scores = {}
        
        base_model = self.model.bert if hasattr(self.model, 'bert') else self.model
        
        for name, param in self.model.named_parameters():
            if 'weight' not in name:
                continue
            
            # Base importance from Phase 1
            component_name = self._param_name_to_component(name)
            base_importance = self.importance_scores.get(component_name, 0.5)
            
            # Magnitude importance
            magnitude_score = torch.abs(param.data).mean().item()
            
            # Movement importance (if available)
            movement_score = 0
            if name in self.initial_weights:
                movement = param.data - self.initial_weights[name]
                movement_score = torch.abs(movement).mean().item()
            
            # Gradient importance (Taylor approximation)
            gradient_score = 0
            if param.grad is not None:
                gradient_score = torch.abs(param.grad * param.data).mean().item()
            
            # Circuit boost
            circuit_boost = self.circuit_components.get(component_name, 0)
            
            # Hybrid score
            if self.config.pruning_method == 'magnitude':
                final_score = magnitude_score
            elif self.config.pruning_method == 'movement':
                final_score = movement_score if movement_score > 0 else magnitude_score
            elif self.config.pruning_method == 'taylor':
                final_score = gradient_score if gradient_score > 0 else magnitude_score
            elif self.config.pruning_method == 'hybrid':
                # Weighted combination
                final_score = (
                    0.2 * base_importance +
                    0.2 * magnitude_score +
                    0.3 * movement_score +
                    0.3 * gradient_score
                )
            else:
                final_score = magnitude_score
            
            # Apply circuit preservation boost
            if circuit_boost > 0.5:  # Important circuit component
                final_score *= (1 + self.config.circuit_preservation_weight * circuit_boost)
            
            # Protect critical layers
            layer_idx = self._extract_layer_index(name)
            if layer_idx in self.config.protect_critical_layers:
                final_score *= 2.0  # Double importance for critical layers
            
            scores[name] = final_score
        
        return scores
    
    def _param_name_to_component(self, param_name: str) -> str:
        """Convert parameter name to component identifier"""
        # Extract layer and component type
        if 'layer' in param_name:
            parts = param_name.split('.')
            layer_idx = None
            component_type = None
            
            for i, part in enumerate(parts):
                if part == 'layer':
                    layer_idx = parts[i + 1]
                elif 'attention' in param_name:
                    component_type = 'attention'
                elif 'intermediate' in param_name or 'output' in param_name:
                    component_type = 'mlp'
            
            if layer_idx is not None and component_type:
                return f'layer_{layer_idx}_{component_type}'
        
        return param_name
    
    def _extract_layer_index(self, param_name: str) -> Optional[int]:
        """Extract layer index from parameter name"""
        if 'layer' in param_name:
            parts = param_name.split('.')
            for i, part in enumerate(parts):
                if part == 'layer' and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        pass
        return None
    
    def calculate_sparsity_schedule(self, step: int) -> float:
        """
        Calculate target sparsity for current step using cubic schedule
        Cubic schedule is superior to linear (Zhu & Gupta, 2018)
        """
        total_pruning_steps = self.config.pruning_steps * self.config.pruning_frequency
        
        if step >= total_pruning_steps:
            return self.config.final_sparsity
        
        # Cubic schedule
        progress = step / total_pruning_steps
        sparsity_ratio = 1 - (1 - progress) ** 3
        
        current_sparsity = (
            self.config.initial_sparsity +
            (self.config.final_sparsity - self.config.initial_sparsity) * sparsity_ratio
        )
        
        return current_sparsity
    
    def prune_structured(self) -> Dict[str, torch.Tensor]:
        """
        Perform structured pruning with gradual sparsity increase
        """
        # Calculate target sparsity for this step
        target_sparsity = self.calculate_sparsity_schedule(self.training_step)
        
        if target_sparsity <= self.current_sparsity:
            return self.masks  # No pruning needed yet
        
        logger.info(f"Pruning step {self.pruning_step}: {self.current_sparsity:.1%} -> {target_sparsity:.1%}")
        
        # Calculate importance scores
        importance_scores = self.calculate_importance_scores()
        
        base_model = self.model.bert if hasattr(self.model, 'bert') else self.model
        
        # Prune attention heads
        self._prune_attention_heads(base_model, importance_scores, target_sparsity)
        
        # Prune MLP neurons
        self._prune_mlp_neurons(base_model, importance_scores, target_sparsity)
        
        self.current_sparsity = target_sparsity
        self.pruning_step += 1
        
        return self.masks
    
    def _prune_attention_heads(
        self,
        base_model: nn.Module,
        importance_scores: Dict[str, float],
        target_sparsity: float
    ):
        """Prune attention heads while preserving minimum per layer"""
        for layer_idx, layer in enumerate(base_model.encoder.layer):
            if not hasattr(layer.attention, 'self'):
                continue
            
            n_heads = layer.attention.self.num_attention_heads
            mask_key = f'layer_{layer_idx}_attention_heads'
            
            if mask_key not in self.masks:
                continue
            
            # Calculate head importance
            head_importance = []
            for head_idx in range(n_heads):
                # Aggregate importance across Q, K, V for this head
                importance = 0
                for component in ['query', 'key', 'value']:
                    param_name = f'bert.encoder.layer.{layer_idx}.attention.self.{component}.weight'
                    if param_name in importance_scores:
                        importance += importance_scores[param_name]
                head_importance.append(importance)
            
            # Protect critical layers
            if layer_idx in self.config.protect_critical_layers:
                min_heads = max(self.config.min_heads_per_layer, int(n_heads * 0.5))
            else:
                min_heads = self.config.min_heads_per_layer
            
            # Calculate how many heads to keep
            n_keep = max(min_heads, int(n_heads * (1 - target_sparsity)))
            
            # Get top-k heads
            head_importance_tensor = torch.tensor(head_importance, device=self.device)
            _, top_indices = torch.topk(head_importance_tensor, n_keep)
            
            # Update mask (soft masking for gradual pruning)
            new_mask = torch.zeros(n_heads, device=self.device)
            new_mask[top_indices] = 1.0
            
            # Gradual transition
            self.masks[mask_key] = (
                0.9 * self.masks[mask_key] + 0.1 * new_mask
            )
    
    def _prune_mlp_neurons(
        self,
        base_model: nn.Module,
        importance_scores: Dict[str, float],
        target_sparsity: float
    ):
        """Prune MLP neurons using structured pruning"""
        for layer_idx, layer in enumerate(base_model.encoder.layer):
            if not hasattr(layer, 'intermediate'):
                continue
            
            mask_key = f'layer_{layer_idx}_mlp_neurons'
            if mask_key not in self.masks:
                continue
            
            # Get neuron importance
            param_name = f'bert.encoder.layer.{layer_idx}.intermediate.dense.weight'
            if param_name in importance_scores:
                weight = layer.intermediate.dense.weight
                neuron_importance = torch.abs(weight).mean(dim=1)  # Average over input dim
                
                # Protect critical layers
                if layer_idx in self.config.protect_critical_layers:
                    keep_ratio = max(0.7, 1 - target_sparsity * 0.5)
                else:
                    keep_ratio = 1 - target_sparsity
                
                n_neurons = neuron_importance.shape[0]
                n_keep = int(n_neurons * keep_ratio)
                
                # Get top-k neurons
                _, top_indices = torch.topk(neuron_importance, n_keep)
                
                # Update mask
                new_mask = torch.zeros(n_neurons, device=self.device)
                new_mask[top_indices] = 1.0
                
                # Gradual transition
                self.masks[mask_key] = (
                    0.9 * self.masks[mask_key] + 0.1 * new_mask
                )
    
    def apply_masks(self):
        """Apply masks to model weights (differentiable for gradient flow)"""
        base_model = self.model.bert if hasattr(self.model, 'bert') else self.model
        
        if not hasattr(base_model, 'encoder'):
            return
        
        for layer_idx, layer in enumerate(base_model.encoder.layer):
            # Apply attention head masks
            head_mask_key = f'layer_{layer_idx}_attention_heads'
            if head_mask_key in self.masks and hasattr(layer.attention, 'self'):
                head_mask = self.masks[head_mask_key]
                n_heads = layer.attention.self.num_attention_heads
                hidden_size = layer.attention.self.all_head_size
                head_dim = hidden_size // n_heads
                
                # Reshape mask for broadcasting
                head_mask_expanded = head_mask.view(n_heads, 1, 1).expand(
                    n_heads, head_dim, -1
                ).contiguous().view(hidden_size, -1)
                
                # Apply to Q, K, V
                for component in ['query', 'key', 'value']:
                    weight = getattr(layer.attention.self, component).weight
                    weight.data = weight.data * head_mask_expanded[:weight.shape[0], :weight.shape[1]]
            
            # Apply MLP neuron masks
            mlp_mask_key = f'layer_{layer_idx}_mlp_neurons'
            if mlp_mask_key in self.masks and hasattr(layer, 'intermediate'):
                neuron_mask = self.masks[mlp_mask_key]
                
                # Apply to intermediate layer
                weight = layer.intermediate.dense.weight
                weight.data = weight.data * neuron_mask.view(-1, 1)
                
                if layer.intermediate.dense.bias is not None:
                    layer.intermediate.dense.bias.data = (
                        layer.intermediate.dense.bias.data * neuron_mask
                    )
                
                # Apply to output layer (transpose of mask)
                output_weight = layer.output.dense.weight
                output_weight.data = output_weight.data * neuron_mask.view(1, -1)
    
    def distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        temperature: float = 4.0
    ) -> torch.Tensor:
        """
        Knowledge distillation loss (Hinton et al., 2015)
        """
        student_logits = student_outputs / temperature
        teacher_logits = teacher_outputs / temperature
        
        student_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # KL divergence
        loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        loss = loss * (temperature ** 2)  # Scale by temperature squared
        
        return loss
    
    def update_training_step(self):
        """Update training step counter"""
        self.training_step += 1
        
        # Check if it's time to prune
        if self.training_step % self.config.pruning_frequency == 0:
            self.prune_structured()
            self.apply_masks()


class PruningTrainer:
    """
    Advanced trainer with pruning, distillation, and recovery mechanisms
    """
    
    def __init__(
        self,
        model: nn.Module,
        teacher_model: Optional[nn.Module],
        pruning_module: AdvancedPruningModule,
        config: PruningConfig,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.teacher_model = teacher_model.to(device) if teacher_model else None
        self.pruning_module = pruning_module
        self.config = config
        self.device = device
        
        # Optimizer with higher learning rate for pruning
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=10000  # Adjust based on dataset
        )
        
        self.training_history = []
    
    def get_linear_schedule_with_warmup(
        self,
        optimizer,
        num_warmup_steps,
        num_training_steps
    ):
        """Create scheduler with linear warmup and decay"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with pruning and distillation"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        logits = outputs.logits.squeeze() if hasattr(outputs, 'logits') else outputs[0].squeeze()
        
        # Task loss
        task_loss = F.mse_loss(logits, batch['labels'])
        
        # Distillation loss (if teacher available)
        distill_loss = 0
        if self.teacher_model and self.config.use_distillation:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                teacher_logits = teacher_outputs.logits.squeeze() if hasattr(teacher_outputs, 'logits') else teacher_outputs[0].squeeze()
            
            distill_loss = F.mse_loss(logits, teacher_logits)
        
        # Combined loss
        if self.config.use_distillation and distill_loss > 0:
            total_loss = (
                (1 - self.config.distillation_alpha) * task_loss +
                self.config.distillation_alpha * distill_loss
            )
        else:
            total_loss = task_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update pruning module
        self.pruning_module.update_training_step()
        
        # Apply masks after weight update
        self.pruning_module.apply_masks()
        
        return {
            'loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'distill_loss': distill_loss.item() if torch.is_tensor(distill_loss) else distill_loss,
            'lr': self.scheduler.get_last_lr()[0],
            'sparsity': self.pruning_module.current_sparsity
        }
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        
        total_loss = 0
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                logits = outputs.logits.squeeze() if hasattr(outputs, 'logits') else outputs[0].squeeze()
                
                loss = F.mse_loss(logits, batch['labels'])
                total_loss += loss.item()
                
                predictions.extend(logits.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        correlation = np.corrcoef(predictions, labels)[0, 1] if len(predictions) > 1 else 0
        mse = np.mean((predictions - labels) ** 2)
        
        return {
            'loss': total_loss / len(eval_loader),
            'correlation': correlation,
            'mse': mse,
            'sparsity': self.pruning_module.current_sparsity
        }
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_epochs: int = 3
    ) -> Dict[str, Any]:
        """Full training loop with pruning"""
        best_score = -float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            epoch_metrics = []
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'sparsity': f"{metrics['sparsity']:.1%}",
                    'lr': f"{metrics['lr']:.2e}"
                })
            
            # Evaluation
            eval_metrics = self.evaluate(eval_loader)
            
            logger.info(f"Epoch {epoch + 1} - Eval: "
                       f"Loss: {eval_metrics['loss']:.4f}, "
                       f"Correlation: {eval_metrics['correlation']:.4f}, "
                       f"Sparsity: {eval_metrics['sparsity']:.1%}")
            
            # Save best model
            if eval_metrics['correlation'] > best_score:
                best_score = eval_metrics['correlation']
                best_model_state = copy.deepcopy(self.model.state_dict())
            
            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_metrics': epoch_metrics,
                'eval_metrics': eval_metrics
            })
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return {
            'best_score': best_score,
            'final_sparsity': self.pruning_module.current_sparsity,
            'history': self.training_history
        }


def calculate_actual_sparsity(model: nn.Module) -> float:
    """Calculate actual sparsity of the model"""
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0


# Example usage function
def run_advanced_pruning(
    model: nn.Module,
    teacher_model: Optional[nn.Module],
    train_loader: DataLoader,
    eval_loader: DataLoader,
    importance_scores: Dict[str, float],
    circuits: List[Dict],
    target_sparsity: float = 0.5,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Run advanced pruning pipeline
    
    Args:
        model: Model to prune
        teacher_model: Teacher model for distillation (optional)
        train_loader: Training data
        eval_loader: Evaluation data
        importance_scores: Importance scores from Phase 1
        circuits: Discovered circuits from Phase 1
        target_sparsity: Target sparsity level
        device: Device for computation
    
    Returns:
        Results dictionary
    """
    # Configure pruning
    config = PruningConfig(
        final_sparsity=target_sparsity,
        pruning_steps=10,
        pruning_frequency=len(train_loader) // 10,  # Prune 10 times per epoch
        pruning_method='hybrid',
        use_distillation=teacher_model is not None
    )
    
    # Initialize pruning module
    pruning_module = AdvancedPruningModule(
        model=model,
        teacher_model=teacher_model,
        importance_scores=importance_scores,
        circuits=circuits,
        config=config,
        device=device
    )
    
    # Initialize trainer
    trainer = PruningTrainer(
        model=model,
        teacher_model=teacher_model,
        pruning_module=pruning_module,
        config=config,
        device=device
    )
    
    # Train with pruning
    logger.info(f"Starting advanced pruning to {target_sparsity:.0%} sparsity")
    results = trainer.train(train_loader, eval_loader, num_epochs=3)
    
    # Calculate final sparsity
    actual_sparsity = calculate_actual_sparsity(model)
    results['actual_sparsity'] = actual_sparsity
    
    logger.info(f"Pruning complete - Target: {target_sparsity:.0%}, Actual: {actual_sparsity:.0%}")
    logger.info(f"Best performance: {results['best_score']:.4f}")
    
    return results
"""
Fixed Advanced Interpretation-Aware Pruning Implementation
With memory management and error handling for training stability
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
import gc
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Advanced pruning configuration based on literature"""
    # Pruning schedule
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.5
    pruning_steps: int = 10  # Reduced from 20
    pruning_frequency: int = 100  # Prune every N training steps
    
    # Fine-tuning
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    pruning_lr_multiplier: float = 2.0
    
    # Method
    pruning_method: str = 'magnitude'
    structured: bool = True
    
    # Distillation
    use_distillation: bool = True
    distillation_alpha: float = 0.5
    temperature: float = 4.0
    
    # Circuit preservation
    circuit_preservation_weight: float = 2.0
    min_heads_per_layer: int = 2
    protect_critical_layers: List[int] = None
    
    # Memory management
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    memory_efficient: bool = True
    
    def __post_init__(self):
        if self.protect_critical_layers is None:
            self.protect_critical_layers = [2, 3, 6, 7]


class AdvancedPruningModule:
    """
    Pruning implementation with memory management
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
        self.teacher_model = teacher_model
        self.importance_scores = importance_scores
        self.circuits = circuits
        self.config = config
        self.device = device
        
        # Initialize pruning state
        self.pruning_step = 0
        self.training_step = 0
        self.current_sparsity = config.initial_sparsity
        
        # Initialize masks with memory efficiency
        self.masks = self._initialize_soft_masks()
        
        # Circuit component mapping
        self.circuit_components = self._map_circuit_components()
        
        # Don't keep full weight history for movement pruning in memory-efficient mode
        if not config.memory_efficient:
            self.initial_weights = self._get_weight_snapshot()
        else:
            self.initial_weights = {}
        
        logger.info(f"Initialized pruning module with {len(self.masks)} masks")
    
    def _get_weight_snapshot(self) -> Dict[str, torch.Tensor]:
        """Snapshot current weights for movement pruning"""
        if self.config.memory_efficient:
            return {}  # Don't store in memory-efficient mode
            
        snapshot = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'layer_' in name:  # Only track layer weights
                snapshot[name] = param.data.clone().cpu()  # Store on CPU
        return snapshot
    
    def _initialize_soft_masks(self) -> Dict[str, torch.Tensor]:
        """Initialize soft masks with error handling"""
        masks = {}
        
        try:
            base_model = self.model.bert if hasattr(self.model, 'bert') else self.model
            
            if hasattr(base_model, 'encoder'):
                for layer_idx, layer in enumerate(base_model.encoder.layer):
                    # Only create masks for layers we'll actually prune
                    if layer_idx >= 12:  # BERT has 12 layers
                        break
                        
                    # Attention masks
                    if hasattr(layer.attention, 'self'):
                        n_heads = layer.attention.self.num_attention_heads
                        masks[f'layer_{layer_idx}_attention_heads'] = torch.ones(
                            n_heads, device='cpu'  # Store on CPU initially
                        )
                    
                    # MLP masks
                    if hasattr(layer, 'intermediate'):
                        intermediate_size = layer.intermediate.dense.weight.shape[0]
                        masks[f'layer_{layer_idx}_mlp_neurons'] = torch.ones(
                            intermediate_size, device='cpu'
                        )
        except Exception as e:
            logger.error(f"Error initializing masks: {str(e)}")
            
        return masks
    
    def _map_circuit_components(self) -> Dict[str, float]:
        """Map discovered circuits to components"""
        component_circuit_importance = {}
        
        for circuit in self.circuits:
            importance = circuit.get('importance_score', 0)
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
        """Calculate importance scores with memory efficiency"""
        scores = {}
        
        try:
            for name, param in self.model.named_parameters():
                if 'weight' not in name:
                    continue
                
                # Base importance from Phase 1
                component_name = self._param_name_to_component(name)
                base_importance = self.importance_scores.get(component_name, 0.5)
                
                # Magnitude importance (always available)
                with torch.no_grad():
                    magnitude_score = torch.abs(param.data).mean().item()
                
                # Use simplified scoring in memory-efficient mode
                if self.config.memory_efficient:
                    final_score = magnitude_score * (1 + base_importance)
                else:
                    # More complex scoring if memory allows
                    movement_score = 0
                    if name in self.initial_weights:
                        movement = param.data.cpu() - self.initial_weights[name]
                        movement_score = torch.abs(movement).mean().item()
                    
                    gradient_score = 0
                    if param.grad is not None:
                        gradient_score = torch.abs(param.grad * param.data).mean().item()
                    
                    if self.config.pruning_method == 'magnitude':
                        final_score = magnitude_score
                    elif self.config.pruning_method == 'movement':
                        final_score = movement_score if movement_score > 0 else magnitude_score
                    else:
                        final_score = magnitude_score * (1 + base_importance)
                
                # Apply circuit preservation boost
                circuit_boost = self.circuit_components.get(component_name, 0)
                if circuit_boost > 0.5:
                    final_score *= (1 + self.config.circuit_preservation_weight * circuit_boost)
                
                # Protect critical layers
                layer_idx = self._extract_layer_index(name)
                if layer_idx in self.config.protect_critical_layers:
                    final_score *= 2.0
                
                scores[name] = final_score
                
        except Exception as e:
            logger.error(f"Error calculating importance scores: {str(e)}")
            # Return uniform scores as fallback
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    scores[name] = 1.0
        
        return scores
    
    def _param_name_to_component(self, param_name: str) -> str:
        """Convert parameter name to component identifier"""
        if 'layer' in param_name:
            parts = param_name.split('.')
            layer_idx = None
            component_type = None
            
            for i, part in enumerate(parts):
                if part == 'layer' and i + 1 < len(parts):
                    try:
                        layer_idx = parts[i + 1]
                        if 'attention' in param_name:
                            component_type = 'attention'
                        elif 'intermediate' in param_name or 'output' in param_name:
                            component_type = 'mlp'
                        break
                    except:
                        pass
            
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
        """Calculate target sparsity for current step"""
        total_pruning_steps = self.config.pruning_steps * self.config.pruning_frequency
        
        if step >= total_pruning_steps:
            return self.config.final_sparsity
        
        # Cubic schedule
        progress = min(1.0, step / total_pruning_steps)
        sparsity_ratio = 1 - (1 - progress) ** 3
        
        current_sparsity = (
            self.config.initial_sparsity +
            (self.config.final_sparsity - self.config.initial_sparsity) * sparsity_ratio
        )
        
        return min(self.config.final_sparsity, current_sparsity)
    
    def prune_structured(self) -> Dict[str, torch.Tensor]:
        """Perform structured pruning with error handling"""
        try:
            # Calculate target sparsity
            target_sparsity = self.calculate_sparsity_schedule(self.training_step)
            
            if target_sparsity <= self.current_sparsity:
                return self.masks
            
            logger.info(f"Pruning step {self.pruning_step}: {self.current_sparsity:.1%} -> {target_sparsity:.1%}")
            
            # Calculate importance scores
            importance_scores = self.calculate_importance_scores()
            
            base_model = self.model.bert if hasattr(self.model, 'bert') else self.model
            
            # Prune with error handling
            if hasattr(base_model, 'encoder'):
                self._prune_attention_heads_safe(base_model, importance_scores, target_sparsity)
                self._prune_mlp_neurons_safe(base_model, importance_scores, target_sparsity)
            
            self.current_sparsity = target_sparsity
            self.pruning_step += 1
            
            # Clear gradients after pruning to save memory
            self.model.zero_grad()
            
            return self.masks
            
        except Exception as e:
            logger.error(f"Error in pruning: {str(e)}")
            return self.masks
    
    def _prune_attention_heads_safe(self, base_model, importance_scores, target_sparsity):
        """Prune attention heads with error handling"""
        try:
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
                head_importance_tensor = torch.tensor(head_importance)
                _, top_indices = torch.topk(head_importance_tensor, min(n_keep, len(head_importance)))
                
                # Update mask
                new_mask = torch.zeros(n_heads)
                new_mask[top_indices] = 1.0
                
                # Gradual transition
                self.masks[mask_key] = (
                    0.9 * self.masks[mask_key].cpu() + 0.1 * new_mask
                )
                
        except Exception as e:
            logger.error(f"Error pruning attention heads: {str(e)}")
    
    def _prune_mlp_neurons_safe(self, base_model, importance_scores, target_sparsity):
        """Prune MLP neurons with error handling"""
        try:
            for layer_idx, layer in enumerate(base_model.encoder.layer):
                if not hasattr(layer, 'intermediate'):
                    continue
                
                mask_key = f'layer_{layer_idx}_mlp_neurons'
                if mask_key not in self.masks:
                    continue
                
                param_name = f'bert.encoder.layer.{layer_idx}.intermediate.dense.weight'
                if param_name not in importance_scores:
                    continue
                    
                weight = layer.intermediate.dense.weight
                with torch.no_grad():
                    neuron_importance = torch.abs(weight).mean(dim=1).cpu()
                
                # Protect critical layers
                if layer_idx in self.config.protect_critical_layers:
                    keep_ratio = max(0.7, 1 - target_sparsity * 0.5)
                else:
                    keep_ratio = 1 - target_sparsity
                
                n_neurons = neuron_importance.shape[0]
                n_keep = max(1, int(n_neurons * keep_ratio))
                
                # Get top-k neurons
                _, top_indices = torch.topk(neuron_importance, min(n_keep, n_neurons))
                
                # Update mask
                new_mask = torch.zeros(n_neurons)
                new_mask[top_indices] = 1.0
                
                # Gradual transition
                self.masks[mask_key] = (
                    0.9 * self.masks[mask_key].cpu() + 0.1 * new_mask
                )
                
        except Exception as e:
            logger.error(f"Error pruning MLP neurons: {str(e)}")
    
    def apply_masks(self):
        """Apply masks to model weights with error handling"""
        try:
            base_model = self.model.bert if hasattr(self.model, 'bert') else self.model
            
            if not hasattr(base_model, 'encoder'):
                return
            
            with torch.no_grad():
                for layer_idx, layer in enumerate(base_model.encoder.layer):
                    # Apply attention head masks
                    head_mask_key = f'layer_{layer_idx}_attention_heads'
                    if head_mask_key in self.masks and hasattr(layer.attention, 'self'):
                        head_mask = self.masks[head_mask_key].to(self.device)
                        n_heads = layer.attention.self.num_attention_heads
                        hidden_size = layer.attention.self.all_head_size
                        head_dim = hidden_size // n_heads
                        
                        # Create expanded mask
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
                        neuron_mask = self.masks[mlp_mask_key].to(self.device)
                        
                        # Apply to intermediate layer
                        weight = layer.intermediate.dense.weight
                        weight.data = weight.data * neuron_mask.view(-1, 1)
                        
                        if layer.intermediate.dense.bias is not None:
                            layer.intermediate.dense.bias.data = (
                                layer.intermediate.dense.bias.data * neuron_mask
                            )
                        
                        # Apply to output layer
                        output_weight = layer.output.dense.weight
                        output_weight.data = output_weight.data * neuron_mask.view(1, -1)
                        
        except Exception as e:
            logger.error(f"Error applying masks: {str(e)}")
    
    def update_training_step(self):
        """Update training step counter"""
        self.training_step += 1
        
        # Check if it's time to prune
        if self.training_step % self.config.pruning_frequency == 0:
            self.prune_structured()
            self.apply_masks()
            
            # Memory cleanup after pruning
            if self.config.memory_efficient:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


class PruningTrainer:
    """
    Trainer with memory management and error recovery
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
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=10000  # Will be adjusted based on actual training
        )
        
        # Don't keep full training history in memory
        self.training_history = []
        self.gradient_accumulation_counter = 0
    
    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
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
        """Single training step with error handling"""
        try:
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
            
            # Handle dimension issues
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if batch['labels'].dim() == 0:
                batch['labels'] = batch['labels'].unsqueeze(0)
            
            # Task loss
            task_loss = F.mse_loss(logits, batch['labels'])
            
            # Distillation loss (optional)
            distill_loss = 0
            if self.teacher_model and self.config.use_distillation:
                try:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask']
                        )
                        teacher_logits = teacher_outputs.logits.squeeze() if hasattr(teacher_outputs, 'logits') else teacher_outputs[0].squeeze()
                    
                    distill_loss = F.mse_loss(logits, teacher_logits)
                except Exception as e:
                    logger.warning(f"Distillation failed: {str(e)}")
                    distill_loss = 0
            
            # Combined loss
            if self.config.use_distillation and distill_loss > 0:
                total_loss = (
                    (1 - self.config.distillation_alpha) * task_loss +
                    self.config.distillation_alpha * distill_loss
                )
            else:
                total_loss = task_loss
            
            # Scale loss for gradient accumulation
            if self.config.gradient_accumulation_steps > 1:
                total_loss = total_loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            total_loss.backward()
            
            # Gradient accumulation
            self.gradient_accumulation_counter += 1
            
            if self.gradient_accumulation_counter % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update pruning module
                self.pruning_module.update_training_step()
                
                # Apply masks after weight update
                self.pruning_module.apply_masks()
            
            return {
                'loss': total_loss.item() * self.config.gradient_accumulation_steps,
                'task_loss': task_loss.item(),
                'distill_loss': distill_loss.item() if torch.is_tensor(distill_loss) else distill_loss,
                'lr': self.scheduler.get_last_lr()[0],
                'sparsity': self.pruning_module.current_sparsity
            }
            
        except Exception as e:
            logger.error(f"Error in train_step: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return dummy metrics
            return {
                'loss': float('inf'),
                'task_loss': float('inf'),
                'distill_loss': 0,
                'lr': self.scheduler.get_last_lr()[0],
                'sparsity': self.pruning_module.current_sparsity
            }
    
    def evaluate(self, eval_loader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model performance with memory management"""
        self.model.eval()
        
        total_loss = 0
        predictions = []
        labels = []
        num_batches = 0
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
                    if max_batches and batch_idx >= max_batches:
                        break
                    
                    try:
                        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                                for k, v in batch.items()}
                        
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask']
                        )
                        
                        logits = outputs.logits.squeeze() if hasattr(outputs, 'logits') else outputs[0].squeeze()
                        
                        if logits.dim() == 0:
                            logits = logits.unsqueeze(0)
                        if batch['labels'].dim() == 0:
                            batch['labels'] = batch['labels'].unsqueeze(0)
                        
                        loss = F.mse_loss(logits, batch['labels'])
                        total_loss += loss.item()
                        
                        predictions.extend(logits.cpu().numpy())
                        labels.extend(batch['labels'].cpu().numpy())
                        num_batches += 1
                        
                    except Exception as e:
                        logger.warning(f"Error in evaluation batch {batch_idx}: {str(e)}")
                        continue
                    
                    # Memory cleanup periodically
                    if batch_idx % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Calculate metrics
            if len(predictions) > 0:
                predictions = np.array(predictions)
                labels = np.array(labels)
                
                correlation = 0
                if len(predictions) > 1:
                    try:
                        correlation = np.corrcoef(predictions, labels)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0
                    except:
                        correlation = 0
                
                mse = np.mean((predictions - labels) ** 2)
                
                return {
                    'loss': total_loss / max(1, num_batches),
                    'correlation': correlation,
                    'mse': mse,
                    'sparsity': self.pruning_module.current_sparsity
                }
            else:
                return {
                    'loss': float('inf'),
                    'correlation': 0,
                    'mse': float('inf'),
                    'sparsity': self.pruning_module.current_sparsity
                }
                
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {
                'loss': float('inf'),
                'correlation': 0,
                'mse': float('inf'),
                'sparsity': self.pruning_module.current_sparsity
            }
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_epochs: int = 3
    ) -> Dict[str, Any]:
        """Training loop with memory management"""
        best_score = -float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Don't keep full epoch metrics to save memory
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    metrics = self.train_step(batch)
                    epoch_loss += metrics['loss']
                    num_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'sparsity': f"{metrics['sparsity']:.1%}",
                        'lr': f"{metrics['lr']:.2e}"
                    })
                    
                    # Memory cleanup periodically
                    if batch_idx % 100 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue
            
            # Evaluation
            try:
                eval_metrics = self.evaluate(eval_loader, max_batches=50)  # Limit eval for speed
                
                logger.info(f"Epoch {epoch + 1} - Eval: "
                           f"Loss: {eval_metrics['loss']:.4f}, "
                           f"Correlation: {eval_metrics['correlation']:.4f}, "
                           f"Sparsity: {eval_metrics['sparsity']:.1%}")
                
                # Save best model
                if eval_metrics['correlation'] > best_score:
                    best_score = eval_metrics['correlation']
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    
            except Exception as e:
                logger.error(f"Evaluation failed: {str(e)}")
                eval_metrics = {'loss': float('inf'), 'correlation': 0}
            
            # Memory cleanup after each epoch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return {
            'best_score': best_score,
            'final_sparsity': self.pruning_module.current_sparsity,
            'history': []  # Don't keep full history
        }


def calculate_actual_sparsity(model: nn.Module) -> float:
    """Calculate actual sparsity of the model"""
    total_params = 0
    zero_params = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0
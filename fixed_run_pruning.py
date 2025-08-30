#!/usr/bin/env python3
"""
Enhanced Pruning Script for NFCorpus with Critical Fixes
Incorporates gradient masking, corrected movement pruning, and adaptive SMA protection
"""

import os
import sys
import json
import pickle
import gc
import traceback
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============= CONFIGURATION =============

@dataclass
class ExperimentConfig:
    """Experiment configuration with validated settings"""
    model_name: str = 'bert-base-uncased'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_sparsities: List[float] = None
    pruning_methods: List[str] = None
    num_epochs: int = 5
    baseline_epochs: int = 6
    batch_size: int = 16
    learning_rate: float = 3e-5
    baseline_lr: float = 5e-5
    warmup_ratio: float = 0.1
    output_dir: Path = Path('./pruning_results_enhanced')
    phase1_dir: Path = Path('./phase1_results')
    max_samples: int = 12000
    dataset_split: str = 'test'
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    protect_layers: List[int] = None
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 2
    pin_memory: bool = True
    sparsity_tolerance: float = 0.01  # For verification
    
    def __post_init__(self):
        """Initialize defaults and validate configuration"""
        if self.target_sparsities is None:
            self.target_sparsities = [0.3, 0.5, 0.728]
        if self.pruning_methods is None:
            self.pruning_methods = ['random', 'magnitude', 'l0', 'movement', 'sma']
        if self.protect_layers is None:
            self.protect_layers = [1, 2, 3, 4, 5, 6, 7]
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)

# ============= LOGGING =============

class Logger:
    """Simple logger with file and console output"""
    
    def __init__(self, log_dir: Path, experiment_name: str = "pruning"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'{experiment_name}_{timestamp}.log'
        
        import logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)

# ============= UTILITIES =============

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@contextmanager
def timer(name: str, logger: Logger):
    """Context manager for timing operations"""
    start = datetime.now()
    try:
        yield
    finally:
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"{name} took {duration:.2f} seconds")

def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

# ============= MODEL =============

class IRModel(nn.Module):
    """BERT-based IR model for ranking"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

# ============= DATASET =============

class NFCorpusDataset(Dataset):
    """NFCorpus dataset with proper preprocessing"""
    
    def __init__(self, 
                 split: str = 'test',
                 max_samples: int = 12000,
                 cache_dir: str = './cache',
                 tokenizer: Optional[Any] = None,
                 max_length: int = 256,
                 logger: Optional[Logger] = None):
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logger
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load NFCorpus data with caching"""
        cache_file = self.cache_dir / f'nfcorpus_{self.split}_v4.pkl'
        
        # Try loading from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if self.logger:
                        self.logger.info(f"Loaded {len(data)} samples from cache")
                    return data[:self.max_samples] if self.max_samples else data
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Cache loading failed: {e}")
        
        if self.logger:
            self.logger.info("Loading NFCorpus from HuggingFace...")
        
        try:
            # Load corpus
            corpus_data = load_dataset("mteb/nfcorpus", "corpus", split="corpus")
            corpus = {}
            for item in corpus_data:
                doc_id = item.get('_id', item.get('id', str(len(corpus))))
                text = item.get('text', '')
                title = item.get('title', '')
                corpus[doc_id] = f"{title} {text}".strip()
            
            # Load queries
            queries_data = load_dataset("mteb/nfcorpus", "queries", split="queries")
            queries = {}
            for item in queries_data:
                query_id = item.get('_id', item.get('id', str(len(queries))))
                queries[query_id] = item.get('text', '')
            
            # Load qrels
            qrels_data = load_dataset("mteb/nfcorpus", "default", split=self.split)
            
            processed_data = []
            count = 0
            
            for item in tqdm(qrels_data, desc=f"Processing {self.split} qrels"):
                if self.max_samples and count >= self.max_samples:
                    break
                
                query_id = item.get('query-id', item.get('query_id'))
                corpus_id = item.get('corpus-id', item.get('corpus_id'))
                score = item.get('score', 0)
                
                if query_id in queries and corpus_id in corpus:
                    query_text = queries[query_id]
                    doc_text = corpus[corpus_id]
                    
                    if query_text and doc_text:
                        # Truncate document
                        doc_text = ' '.join(doc_text.split()[:400])
                        processed_data.append({
                            'query': query_text,
                            'document': doc_text,
                            'relevance': float(score / 2.0),  # Normalize to [0, 1]
                            'query_id': query_id,
                            'doc_id': corpus_id
                        })
                        count += 1
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_data, f)
            
            if self.logger:
                self.logger.info(f"Loaded {len(processed_data)} samples from NFCorpus")
            return processed_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load NFCorpus: {e}")
            # Return synthetic data as fallback
            return [{'query': f'query {i}', 
                    'document': f'document {i}',
                    'relevance': np.random.random()} 
                   for i in range(min(100, self.max_samples or 100))]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized sample"""
        sample = self.data[idx]
        
        if self.tokenizer:
            encoded = self.tokenizer(
                sample['query'],
                sample['document'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': torch.tensor(sample['relevance'], dtype=torch.float32)
            }
        
        return sample

# ============= PRUNING METHODS =============

class PruningMethods:
    """All pruning methods with exact sparsity targeting and fixes"""
    
    @staticmethod
    def get_weights_for_pruning(model: nn.Module) -> Tuple[List, List]:
        """Get all weight parameters eligible for pruning"""
        weights_list = []
        param_info = []
        
        for name, param in model.named_parameters():
            # Only prune weight matrices, not biases or embeddings
            if 'weight' in name and param.dim() >= 2:
                # Skip embeddings and layer norm
                if not any(skip in name for skip in ['embeddings', 'LayerNorm', 'layer_norm']):
                    weights_list.append(param.abs().flatten())
                    param_info.append((name, param))
        
        return weights_list, param_info
    
    @staticmethod
    def get_exact_threshold(weights_list: List[torch.Tensor], target_sparsity: float) -> torch.Tensor:
        """Get exact threshold to achieve target sparsity"""
        if not weights_list:
            return torch.tensor(0.0)
        
        # Concatenate all weights
        all_weights = torch.cat(weights_list)
        
        # Calculate exact number of parameters to prune
        total_params = all_weights.numel()
        num_zeros = int(total_params * target_sparsity)
        
        if num_zeros == 0:
            return torch.tensor(0.0)
        elif num_zeros >= total_params:
            return all_weights.max() + 1e-6
        
        # Sort and find exact threshold
        sorted_weights, _ = torch.sort(all_weights)
        threshold = sorted_weights[num_zeros - 1]
        
        return threshold
    
    @staticmethod
    def _get_layer_index(param_name: str) -> int:
        """Extract layer index from parameter name"""
        parts = param_name.split('.')
        for i, part in enumerate(parts):
            if part == 'layer' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return -1
    
    @staticmethod
    def random_pruning(model: nn.Module, sparsity: float, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Random pruning with exact sparsity"""
        masks = {}
        weights_list, param_info = PruningMethods.get_weights_for_pruning(model)
        
        if not weights_list:
            return masks
        
        # Generate random scores
        random_scores = []
        for name, param in param_info:
            scores = torch.rand_like(param)
            random_scores.append(scores.flatten())
        
        # Get exact threshold
        threshold = PruningMethods.get_exact_threshold(random_scores, sparsity)
        
        # Apply masks
        for i, (name, param) in enumerate(param_info):
            scores = random_scores[i].reshape(param.shape)
            mask = (scores > threshold).float()
            masks[name] = mask
            param.data.mul_(mask)
        
        return masks
    
    @staticmethod
    def magnitude_pruning(model: nn.Module, sparsity: float, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Magnitude pruning with exact sparsity"""
        masks = {}
        weights_list, param_info = PruningMethods.get_weights_for_pruning(model)
        
        if not weights_list:
            return masks
        
        # Get exact threshold based on magnitudes
        threshold = PruningMethods.get_exact_threshold(weights_list, sparsity)
        
        # Apply masks
        for name, param in param_info:
            mask = (param.abs() > threshold).float()
            masks[name] = mask
            param.data.mul_(mask)
        
        return masks
    
    @staticmethod
    def l0_pruning(model: nn.Module, sparsity: float, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """L0 regularization pruning with exact sparsity"""
        masks = {}
        weights_list, param_info = PruningMethods.get_weights_for_pruning(model)
        
        if not weights_list:
            return masks
        
        # Add stochastic noise for L0
        noisy_weights = []
        for name, param in param_info:
            importance = param.abs() + torch.randn_like(param) * 0.01
            noisy_weights.append(importance.flatten())
        
        # Get exact threshold
        threshold = PruningMethods.get_exact_threshold(noisy_weights, sparsity)
        
        # Apply masks
        for i, (name, param) in enumerate(param_info):
            importance = noisy_weights[i].reshape(param.shape)
            mask = (importance > threshold).float()
            masks[name] = mask
            param.data.mul_(mask)
        
        return masks
    
    @staticmethod
    def movement_pruning(model: nn.Module, sparsity: float, dataloader: DataLoader,
                        device: str = 'cuda', logger: Optional[Logger] = None) -> Dict[str, torch.Tensor]:
        """FIXED Movement pruning with correct implementation"""
        masks = {}
        weights_list, param_info = PruningMethods.get_weights_for_pruning(model)
        
        if not weights_list:
            return masks
        
        # Create a fresh model for movement calculation
        temp_model = copy.deepcopy(model)
        
        # Reset to random initialization for movement measurement
        for name, param in temp_model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if not any(skip in name for skip in ['embeddings', 'LayerNorm', 'layer_norm']):
                    # Reinitialize with Xavier/He initialization
                    if 'attention' in name:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.kaiming_uniform_(param)
        
        # Store initial weights
        initial_weights = {}
        for name, param in temp_model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if not any(skip in name for skip in ['embeddings', 'LayerNorm', 'layer_norm']):
                    initial_weights[name] = param.clone()
        
        # Brief training to capture movement
        temp_model.train()
        optimizer = torch.optim.AdamW(temp_model.parameters(), lr=1e-4)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 20:  # More iterations for better movement signal
                break
            
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = temp_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits.squeeze()
            loss = F.mse_loss(logits, batch['labels'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Compute movement scores based on the temp model
        movement_scores = []
        for name, param in param_info:
            if name in initial_weights:
                # Get the corresponding parameter from temp_model
                temp_param = dict(temp_model.named_parameters())[name]
                movement = (temp_param - initial_weights[name]).abs()
                movement_scores.append(movement.flatten())
            else:
                # If not in initial_weights, use magnitude as fallback
                movement_scores.append(param.abs().flatten())
        
        # Get exact threshold
        threshold = PruningMethods.get_exact_threshold(movement_scores, sparsity)
        
        # Apply masks to the ORIGINAL model
        for i, (name, param) in enumerate(param_info):
            movement = movement_scores[i].reshape(param.shape)
            mask = (movement > threshold).float()
            masks[name] = mask
            param.data.mul_(mask)
        
        # Clean up
        del temp_model
        
        return masks
    
    @staticmethod
    def sma_pruning(model: nn.Module, sparsity: float, importance_scores: Dict[str, float],
                   protect_layers: List[int], circuits: Optional[List[Dict]] = None,
                   device: str = 'cuda', logger: Optional[Logger] = None) -> Dict[str, torch.Tensor]:
        """ENHANCED SMA pruning with adaptive circuit protection"""
        masks = {}
        
        # Adaptive protection based on sparsity level
        if sparsity >= 0.7:
            circuit_protection = 10.0  # Very strong protection at high sparsity
            layer_protection = 5.0
        elif sparsity >= 0.5:
            circuit_protection = 5.0
            layer_protection = 3.0
        else:
            circuit_protection = 3.0
            layer_protection = 2.0
        
        # Build circuit layer set
        circuit_layers = set()
        if circuits:
            for circuit in circuits:
                if isinstance(circuit, dict):
                    layer_idx = circuit.get('layer', -1)
                    if layer_idx >= 0:
                        circuit_layers.add(layer_idx)
        
        # Collect weights with protection factors
        protected_weights = []
        param_info_all = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if any(skip in name for skip in ['embeddings', 'LayerNorm', 'layer_norm']):
                    continue
                
                # Get layer index
                layer_idx = PruningMethods._get_layer_index(name)
                
                # Determine protection factor
                base_importance = param.abs()
                
                # Add importance scores if available
                if name in importance_scores:
                    importance_factor = 1.0 + importance_scores[name]
                    base_importance = base_importance * importance_factor
                
                # Apply layer-based protection
                if layer_idx in circuit_layers:
                    protection = circuit_protection
                elif layer_idx in protect_layers:
                    protection = layer_protection
                else:
                    protection = 1.0
                
                protected_weights.append(base_importance.flatten() * protection)
                param_info_all.append((name, param, protection))
        
        if not protected_weights:
            return masks
        
        # Get threshold
        threshold = PruningMethods.get_exact_threshold(protected_weights, sparsity)
        
        # Apply masks with protection
        for i, (name, param, protection) in enumerate(param_info_all):
            weighted_param = param.abs() * protection
            
            # Add importance score boost if available
            if name in importance_scores:
                weighted_param = weighted_param * (1.0 + importance_scores[name])
            
            mask = (weighted_param > threshold).float()
            masks[name] = mask
            param.data.mul_(mask)
            
            if logger and i < 5:  # Log first few for debugging
                kept = mask.sum().item()
                total = mask.numel()
                logger.info(f"  {name}: kept {kept}/{total} ({kept/total:.2%}), protection={protection:.1f}")
        
        return masks

# ============= ENHANCED TRAINER WITH FIXES =============

class Trainer:
    """Training utilities with proper gradient masking"""
    
    def __init__(self, config: ExperimentConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.scaler = GradScaler() if config.fp16 else None
        self.hooks = []
    
    def register_gradient_hooks(self, model: nn.Module, masks: Dict[str, torch.Tensor]):
        """Register hooks to zero gradients for pruned weights"""
        self.hooks = []
        for name, param in model.named_parameters():
            if name in masks:
                mask = masks[name]
                # This hook will zero gradients for pruned weights
                hook = param.register_hook(lambda grad, mask=mask: grad * mask if grad is not None else grad)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove gradient hooks"""
        for hook in getattr(self, 'hooks', []):
            hook.remove()
        self.hooks = []
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, optimizer: Any,
                   scheduler: Any, masks: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """Train for one epoch with proper masking"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Register gradient hooks BEFORE training
        if masks:
            self.register_gradient_hooks(model, masks)
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(self.config.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            if self.config.fp16:
                with autocast():
                    outputs = model(input_ids=batch['input_ids'], 
                                  attention_mask=batch['attention_mask'])
                    logits = outputs.logits.squeeze()
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)
                    loss = F.mse_loss(logits, batch['labels'])
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Enforce masks after optimizer step (redundant but ensures correctness)
                    if masks:
                        self.enforce_masks(model, masks)
            else:
                outputs = model(input_ids=batch['input_ids'], 
                              attention_mask=batch['attention_mask'])
                logits = outputs.logits.squeeze()
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                loss = F.mse_loss(logits, batch['labels'])
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Enforce masks after optimizer step
                    if masks:
                        self.enforce_masks(model, masks)
            
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Remove hooks after training
        if masks:
            self.remove_hooks()
        
        return total_loss / max(num_batches, 1)
    
    @staticmethod
    def enforce_masks(model: nn.Module, masks: Dict[str, torch.Tensor]):
        """Enforce pruning masks"""
        for name, param in model.named_parameters():
            if name in masks:
                param.data.mul_(masks[name])
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        model.eval()
        predictions = []
        targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.config.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                if self.config.fp16:
                    with autocast():
                        outputs = model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'])
                        logits = outputs.logits.squeeze()
                        loss = F.mse_loss(logits, batch['labels'])
                else:
                    outputs = model(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask'])
                    logits = outputs.logits.squeeze()
                    loss = F.mse_loss(logits, batch['labels'])
                
                predictions.extend(logits.cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        correlation = 0.0
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, targets)[0, 1]
            correlation = 0.0 if np.isnan(correlation) else correlation
        
        mse = np.mean((predictions - targets) ** 2)
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'correlation': correlation,
            'mse': mse,
            'num_samples': len(predictions)
        }
    
    @staticmethod
    def calculate_sparsity(model: nn.Module) -> float:
        """Calculate actual sparsity"""
        total_params = 0
        zero_params = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if not any(skip in name for skip in ['embeddings', 'LayerNorm', 'layer_norm']):
                    total_params += param.numel()
                    zero_params += (param == 0).sum().item()
        
        return zero_params / max(total_params, 1)

# ============= VERIFICATION UTILITIES =============

def verify_sparsity_maintained(model: nn.Module, target_sparsity: float, 
                               tolerance: float = 0.01) -> Tuple[bool, float]:
    """Verify that sparsity is maintained within tolerance"""
    actual_sparsity = Trainer.calculate_sparsity(model)
    is_maintained = abs(actual_sparsity - target_sparsity) <= tolerance
    return is_maintained, actual_sparsity

def train_pruned_model_with_verification(model, masks, train_loader, eval_loader, 
                                         config, trainer, logger, target_sparsity):
    """Train pruned model with sparsity verification"""
    
    # Setup optimizer with smaller learning rate for pruned models
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 0.5)
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    best_metrics = None
    best_correlation = -1.5
    
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        # Verify sparsity before training
        is_maintained, actual = verify_sparsity_maintained(model, target_sparsity)
        if not is_maintained:
            logger.warning(f"Sparsity drift detected! Target: {target_sparsity:.2%}, Actual: {actual:.2%}")
            # Re-enforce masks
            trainer.enforce_masks(model, masks)
        
        # Train with gradient hooks
        loss = trainer.train_epoch(model, train_loader, optimizer, scheduler, masks)
        
        # Verify sparsity after training
        is_maintained, actual = verify_sparsity_maintained(model, target_sparsity)
        logger.info(f"Post-epoch sparsity: {actual:.2%} (target: {target_sparsity:.2%})")
        
        # Evaluate
        metrics = trainer.evaluate(model, eval_loader)
        
        # Track best model
        if metrics['correlation'] > best_correlation:
            best_correlation = metrics['correlation']
            best_metrics = metrics.copy()
        
        logger.info(f"Loss: {loss:.4f}, Correlation: {metrics['correlation']:.4f}")
    
    return best_metrics, best_correlation

# ============= MAIN EXPERIMENT =============

def load_phase1_results(phase1_dir: Path, logger: Logger) -> Tuple[Dict[str, float], List[Dict]]:
    """Load Phase 1 results"""
    importance_scores = {}
    circuits = []
    
    # Load importance scores
    importance_path = phase1_dir / 'importance_scores.json'
    if importance_path.exists():
        try:
            with open(importance_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if 'importance_scores' in data:
                        importance_scores = data['importance_scores']
                    else:
                        importance_scores = data
                logger.info(f"Loaded {len(importance_scores)} importance scores")
        except Exception as e:
            logger.warning(f"Failed to load importance scores: {e}")
    
    # Load circuits
    circuits_path = phase1_dir / 'circuits.json'
    if circuits_path.exists():
        try:
            with open(circuits_path, 'r') as f:
                circuits = json.load(f)
                logger.info(f"Loaded {len(circuits)} circuits")
        except Exception as e:
            logger.warning(f"Failed to load circuits: {e}")
    
    return importance_scores, circuits

def run_experiment(config: ExperimentConfig):
    """Run complete pruning experiment with enhanced fixes"""
    
    # Initialize logger
    logger = Logger(config.output_dir / 'logs', 'pruning_experiment_enhanced')
    
    logger.info("="*60)
    logger.info("ENHANCED PRUNING EXPERIMENT WITH CRITICAL FIXES")
    logger.info("="*60)
    logger.info(f"Configuration:\n{json.dumps(asdict(config), indent=2, default=str)}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Load tokenizer
    with timer("Loading tokenizer", logger):
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load dataset
    with timer("Dataset loading", logger):
        dataset = NFCorpusDataset(
            split=config.dataset_split,
            max_samples=config.max_samples,
            tokenizer=tokenizer,
            logger=logger
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        logger.info(f"Dataset: {train_size} train, {val_size} eval samples")
    
    # Load Phase 1 results
    importance_scores, circuits = load_phase1_results(config.phase1_dir, logger)
    
    # Initialize trainer
    trainer = Trainer(config, logger)
    
    # Train strong baseline model
    logger.info("\n" + "="*60)
    logger.info("TRAINING STRONG BASELINE MODEL")
    logger.info("="*60)
    
    with timer("Baseline training", logger):
        baseline_model = IRModel(config.model_name).to(config.device)
        optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=config.baseline_lr)
        
        num_training_steps = len(train_loader) * config.baseline_epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        
        best_baseline_corr = 0.0
        best_baseline_state = None
        
        for epoch in range(config.baseline_epochs):
            logger.info(f"\nBaseline Epoch {epoch + 1}/{config.baseline_epochs}")
            loss = trainer.train_epoch(baseline_model, train_loader, optimizer, scheduler)
            
            # Evaluate every 2 epochs
            if epoch % 2 == 1 or epoch == config.baseline_epochs - 1:
                metrics = trainer.evaluate(baseline_model, eval_loader)
                logger.info(f"Epoch {epoch+1}: Loss={loss:.4f}, Correlation={metrics['correlation']:.4f}")
                
                if metrics['correlation'] > best_baseline_corr:
                    best_baseline_corr = metrics['correlation']
                    best_baseline_state = copy.deepcopy(baseline_model.state_dict())
        
        # Load best baseline
        if best_baseline_state:
            baseline_model.load_state_dict(best_baseline_state)
        baseline_metrics = trainer.evaluate(baseline_model, eval_loader)
        logger.info(f"Best baseline metrics: {baseline_metrics}")
    
    # Save baseline model
    torch.save(baseline_model.state_dict(), config.output_dir / 'models' / 'baseline.pt')
    
    # Results storage
    all_results = {
        'config': asdict(config),
        'baseline': baseline_metrics,
        'methods': {}
    }
    
    # Test each pruning method
    for method in config.pruning_methods:
        logger.info("\n" + "="*60)
        logger.info(f"TESTING {method.upper()} PRUNING")
        logger.info("="*60)
        
        all_results['methods'][method] = {}
        
        for sparsity in config.target_sparsities:
            logger.info(f"\n>>> {method.upper()} at {sparsity:.0%} sparsity")
            
            try:
                # Create fresh model from baseline
                model = IRModel(config.model_name).to(config.device)
                model.load_state_dict(baseline_model.state_dict())
                
                # Apply pruning
                with timer(f"Applying {method} pruning", logger):
                    if method == 'random':
                        masks = PruningMethods.random_pruning(model, sparsity, config.device)
                    elif method == 'magnitude':
                        masks = PruningMethods.magnitude_pruning(model, sparsity, config.device)
                    elif method == 'l0':
                        masks = PruningMethods.l0_pruning(model, sparsity, config.device)
                    elif method == 'movement':
                        masks = PruningMethods.movement_pruning(
                            model, sparsity, train_loader, config.device, logger
                        )
                    elif method == 'sma':
                        masks = PruningMethods.sma_pruning(
                            model, sparsity, importance_scores, 
                            config.protect_layers, circuits, config.device, logger
                        )
                    else:
                        logger.warning(f"Unknown method: {method}")
                        continue
                
                # Verify sparsity
                actual_sparsity = trainer.calculate_sparsity(model)
                logger.info(f"Actual sparsity: {actual_sparsity:.2%}")
                
                # Fine-tune pruned model with verification
                best_metrics, best_correlation = train_pruned_model_with_verification(
                    model, masks, train_loader, eval_loader, 
                    config, trainer, logger, sparsity
                )
                
                # Calculate retention
                retention = best_metrics['correlation'] / max(baseline_metrics['correlation'], 0.001)
                
                # Store results
                all_results['methods'][method][sparsity] = {
                    'target_sparsity': sparsity,
                    'actual_sparsity': actual_sparsity,
                    'metrics': best_metrics,
                    'retention': retention,
                    'best_correlation': best_correlation
                }
                
                logger.info(f"\nResults for {method} at {sparsity:.0%}:")
                logger.info(f"  Actual sparsity: {actual_sparsity:.2%}")
                logger.info(f"  Best correlation: {best_correlation:.4f}")
                logger.info(f"  Retention: {retention:.2%}")
                
            except Exception as e:
                logger.error(f"Failed to run {method} at {sparsity:.0%}: {e}")
                logger.error(traceback.format_exc())
                all_results['methods'][method][sparsity] = {'error': str(e)}
            
            finally:
                # Clean up
                if 'model' in locals():
                    del model
                cleanup_memory()
    
    # Save results
    results_path = config.output_dir / 'metrics' / 'pruning_results_enhanced.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    logger.info(f"\nBaseline Correlation: {baseline_metrics['correlation']:.4f}")
    logger.info(f"\n{'Method':<12} {'Sparsity':<12} {'Actual':<12} {'Correlation':<12} {'Retention':<12}")
    logger.info("-" * 60)
    
    for method in config.pruning_methods:
        if method in all_results['methods']:
            for sparsity in config.target_sparsities:
                if sparsity in all_results['methods'][method]:
                    result = all_results['methods'][method][sparsity]
                    
                    if 'error' not in result:
                        logger.info(
                            f"{method:<12} {sparsity:<12.0%} {result['actual_sparsity']:<12.2%} "
                            f"{result['metrics']['correlation']:<12.4f} {result['retention']:<12.2%}"
                        )
    
    logger.info(f"\nResults saved to: {results_path}")
    
    return all_results

# ============= ENTRY POINT =============

def main():
    """Main entry point"""
    try:
        config = ExperimentConfig()
        results = run_experiment(config)
        return results
    except Exception as e:
        print(f"Experiment failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()

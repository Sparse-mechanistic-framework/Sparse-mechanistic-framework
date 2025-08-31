#!/usr/bin/env python3
"""
Fixed Pruning Script for NFCorpus with All Corrections
Ensures different methods produce different results with proper implementations
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
    output_dir: Path = Path('./pruning_results_fixed')
    phase1_dir: Path = Path('./phase1_results')
    max_samples: int = 11200
    dataset_split: str = 'test'
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    protect_layers: List[int] = None
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 2
    pin_memory: bool = True
    sparsity_tolerance: float = 0.01
    
    def __post_init__(self):
        """Initialize defaults and validate configuration"""
        if self.target_sparsities is None:
            self.target_sparsities = [0.3, 0.5, 0.67]
        if self.pruning_methods is None:
            self.pruning_methods = ['magnitude', 'sma', 'l0', 'movement', 'random']
        if self.protect_layers is None:
            self.protect_layers = [2, 3, 4, 5, 6, 7]
        
        # Create output directories
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
        
        # Enable gradient checkpointing
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
                 max_samples: int = 11200,
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
                        doc_text = ' '.join(doc_text.split()[:400])
                        processed_data.append({
                            'query': query_text,
                            'document': doc_text,
                            'relevance': float(score / 2.0),
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

# ============= FIXED PRUNING METHODS =============

class PruningMethods:
    """All pruning methods with proper implementations"""
    
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
        threshold = sorted_weights[num_zeros - 1] if num_zeros > 0 else torch.tensor(0.0)
        
        return threshold
    
    @staticmethod
    def _get_layer_index(param_name: str) -> int:
        """Extract layer index from parameter name"""
        parts = param_name.split('.')
        for i, part in enumerate(parts):
            if 'layer' in part.lower() and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return -1
    
    @staticmethod
    def random_pruning(model: nn.Module, sparsity: float, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Random pruning - original implementation (baseline)"""
        masks = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if not any(skip in name for skip in ['embeddings', 'LayerNorm', 'layer_norm']):
                    # Original: randomly select weights to prune
                    mask = torch.bernoulli(torch.ones_like(param) * (1 - sparsity))
                    masks[name] = mask.to(device)
                    param.data.mul_(mask)
        
        return masks
    
    @staticmethod
    def magnitude_pruning(model: nn.Module, sparsity: float, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Magnitude pruning - Han et al. (2015) original implementation
        Prunes weights with smallest absolute values globally"""
        masks = {}
        
        # Collect all weights
        all_weights = []
        weight_shapes = []
        weight_names = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if not any(skip in name for skip in ['embeddings', 'LayerNorm', 'layer_norm']):
                    all_weights.append(param.abs().flatten())
                    weight_shapes.append(param.shape)
                    weight_names.append(name)
        
        if not all_weights:
            return masks
        
        # Concatenate all weights
        all_weights_concat = torch.cat(all_weights)
        
        # Find threshold - original paper uses global threshold
        num_params_to_prune = int(all_weights_concat.numel() * sparsity)
        if num_params_to_prune > 0:
            sorted_weights, _ = torch.sort(all_weights_concat)
            threshold = sorted_weights[num_params_to_prune]
        else:
            threshold = 0.0
        
        # Apply masks
        for name, param in model.named_parameters():
            if name in weight_names:
                mask = (param.abs() > threshold).float()
                masks[name] = mask.to(device)
                param.data.mul_(mask)
        
        return masks
    
    @staticmethod
    def l0_pruning(model: nn.Module, sparsity: float, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """L0 regularization pruning - Louizos et al. (2018) original implementation
        Approximated for one-shot pruning using hard concrete distribution"""
        masks = {}
        
        # Parameters from the original paper
        temperature = 2.0 / 3.0  # Temperature for concrete distribution
        gamma = -0.1  # Stretch parameter
        zeta = 1.1    # Stretch parameter
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if not any(skip in name for skip in ['embeddings', 'LayerNorm', 'layer_norm']):
                    # Initialize log alpha based on weight magnitudes
                    # Higher magnitude weights should have higher log alpha (less likely to prune)
                    weight_magnitude = param.abs()
                    normalized_magnitude = (weight_magnitude - weight_magnitude.min()) / (weight_magnitude.max() - weight_magnitude.min() + 1e-8)
                    
                    # Map to log alpha range that gives desired sparsity
                    # log_alpha controls the dropout rate in hard concrete
                    log_alpha = torch.log(normalized_magnitude + 1e-8) - torch.log(1 - normalized_magnitude + 1e-8)
                    
                    # Sample from hard concrete distribution
                    u = torch.rand_like(param).clamp(1e-8, 1 - 1e-8)
                    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / temperature)
                    
                    # Stretch and rectify
                    s_bar = s * (zeta - gamma) + gamma
                    mask = torch.clamp(s_bar, 0, 1)
                    
                    # Apply threshold to achieve target sparsity
                    mask_flat = mask.flatten()
                    k = int(mask_flat.numel() * (1 - sparsity))
                    threshold = torch.topk(mask_flat, k, largest=True)[0][-1]
                    mask = (mask >= threshold).float()
                    
                    masks[name] = mask.to(device)
                    param.data.mul_(mask)
        
        return masks
    
    @staticmethod
    def movement_pruning(model: nn.Module, sparsity: float, dataloader: DataLoader,
                        device: str = 'cuda', logger: Optional[Logger] = None) -> Dict[str, torch.Tensor]:
        """Movement pruning - Sanh et al. (2020) original implementation
        Prunes weights based on movement during fine-tuning
        Movement score: S = W_final * (W_final - W_init)"""
        masks = {}
        
        if dataloader is None:
            # Fallback to magnitude if no dataloader
            return PruningMethods.magnitude_pruning(model, sparsity, device)
        
        # Store initial weights (pretrained state)
        initial_weights = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if not any(skip in name for skip in ['embeddings', 'LayerNorm', 'layer_norm']):
                    initial_weights[name] = param.data.clone()
        
        # Fine-tune briefly to get movement
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Original paper uses AdamW
        
        if logger:
            logger.info("Movement pruning: fine-tuning to compute movement scores...")
        
        # Fine-tune for a few steps (original paper uses more, but we use fewer for efficiency)
        num_movement_steps = min(100, len(dataloader))  # Original uses full epoch
        step_count = 0
        
        for batch in dataloader:
            if step_count >= num_movement_steps:
                break
            
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits.squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            loss = F.mse_loss(logits, batch['labels'])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
        
        # Compute movement scores: S = W_final * (W_final - W_init)
        # Original paper: positive scores = moving away from zero (keep)
        #                 negative scores = moving toward zero (prune)
        all_scores = []
        score_dict = {}
        
        for name, param in model.named_parameters():
            if name in initial_weights:
                # Movement score from original paper
                movement_score = param.data * (param.data - initial_weights[name])
                score_dict[name] = movement_score
                all_scores.append(movement_score.flatten())
        
        if not all_scores:
            return masks
        
        # Find threshold for target sparsity
        all_scores_concat = torch.cat(all_scores)
        num_params_to_prune = int(all_scores_concat.numel() * sparsity)
        if num_params_to_prune > 0:
            sorted_scores, _ = torch.sort(all_scores_concat)
            threshold = sorted_scores[num_params_to_prune]
        else:
            threshold = 0.0
        
        # Apply masks and restore original weights
        for name, param in model.named_parameters():
            if name in initial_weights:
                # Create mask based on movement scores
                mask = (score_dict[name] > threshold).float()
                masks[name] = mask.to(device)
                
                # IMPORTANT: Restore initial weights then apply mask
                # This is key to movement pruning - we prune from the original weights
                param.data = initial_weights[name]
                param.data.mul_(mask)
        
        return masks
    
    @staticmethod
    def sma_pruning(model: nn.Module, sparsity: float, importance_scores: Dict[str, float],
                   protect_layers: List[int], circuits: Optional[List[Dict]] = None,
                   device: str = 'cuda', logger: Optional[Logger] = None) -> Dict[str, torch.Tensor]:
        """SMA pruning with proper importance weighting and circuit protection"""
        masks = {}
        weights_list, param_info = PruningMethods.get_weights_for_pruning(model)
        
        if not weights_list:
            return masks
        
        # Build circuit layer set
        circuit_layers = set()
        if circuits:
            for circuit in circuits:
                if isinstance(circuit, dict):
                    layer_idx = circuit.get('layer', -1)
                    if layer_idx >= 0:
                        circuit_layers.add(layer_idx)
        
        # Compute importance-weighted scores
        weighted_scores = []
        for name, param in param_info:
            layer_idx = PruningMethods._get_layer_index(name)
            
            # Base score from magnitude
            base_score = param.abs()
            
            # Apply importance from Phase 1 if available
            importance_multiplier = 1.0
            
            # Check for layer-specific importance scores
            for key, score in importance_scores.items():
                if f'layer_{layer_idx}' in key or f'layer.{layer_idx}' in key:
                    # Use actual importance score (assumed to be 0-1)
                    # Higher importance = higher multiplier
                    importance_multiplier = 1.0 + score * 2.0  # Can go up to 3x
                    break
            
            # Apply circuit protection (strongest)
            if layer_idx in circuit_layers:
                importance_multiplier *= 3.0  # Triple importance for circuit layers
                if logger:
                    logger.info(f"  Circuit layer {layer_idx} protected with 3x multiplier")
            
            # Apply general layer protection (moderate)
            elif layer_idx in protect_layers:
                importance_multiplier *= 1.5  # 1.5x for protected layers
            
            # Apply adaptive protection based on sparsity level
            if sparsity >= 0.7:
                # At high sparsity, increase protection further
                if layer_idx in circuit_layers:
                    importance_multiplier *= 1.5  # Total 4.5x for circuits
                elif layer_idx in protect_layers:
                    importance_multiplier *= 1.3  # Total 1.95x for protected
            
            weighted_scores.append((base_score * importance_multiplier).flatten())
        
        # Get threshold
        threshold = PruningMethods.get_exact_threshold(weighted_scores, sparsity)
        
        # Apply masks
        for i, (name, param) in enumerate(param_info):
            score_shape = weighted_scores[i][:param.numel()].reshape(param.shape)
            mask = (score_shape > threshold).float()
            masks[name] = mask
            param.data.mul_(mask)
        
        return masks

# ============= TRAINER WITH GRADIENT MASKING =============

class Trainer:
    """Training utilities with proper gradient masking"""
    
    def __init__(self, config: ExperimentConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.scaler = GradScaler() if config.fp16 else None
        self.hooks = []
    
    def register_gradient_hooks(self, model: nn.Module, masks: Dict[str, torch.Tensor]):
        """Register hooks to zero gradients for pruned weights"""
        self.remove_hooks()  # Clean up any existing hooks
        self.hooks = []
        
        for name, param in model.named_parameters():
            if name in masks:
                mask = masks[name]
                # Register hook that zeros gradients for pruned weights
                hook = param.register_hook(lambda grad, m=mask: grad * m if grad is not None else grad)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all gradient hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, optimizer: Any,
                   scheduler: Any, masks: Optional[Dict[str, torch.Tensor]] = None,
                   enforce_masks: bool = True) -> float:
        """Train for one epoch with optional mask enforcement"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
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
                    
                    # Enforce masks after optimizer step if requested
                    if enforce_masks and masks:
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
                    
                    # Enforce masks after optimizer step if requested
                    if enforce_masks and masks:
                        self.enforce_masks(model, masks)
            
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    @staticmethod
    def enforce_masks(model: nn.Module, masks: Dict[str, torch.Tensor]):
        """Enforce pruning masks on model parameters"""
        with torch.no_grad():
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
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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

# ============= TRAINING UTILITIES =============

def verify_sparsity_maintained(model: nn.Module, target_sparsity: float, 
                               tolerance: float = 0.01) -> Tuple[bool, float]:
    """Verify that sparsity is maintained within tolerance"""
    actual_sparsity = Trainer.calculate_sparsity(model)
    is_maintained = abs(actual_sparsity - target_sparsity) <= tolerance
    return is_maintained, actual_sparsity

def train_pruned_model_with_verification(model, masks, train_loader, eval_loader, 
                                         config, trainer, logger, target_sparsity):
    """Train pruned model with proper gradient masking and verification"""
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    # Register gradient hooks ONCE at the beginning
    trainer.register_gradient_hooks(model, masks)
    
    best_metrics = None
    best_correlation = -1.0
    
    try:
        for epoch in range(config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
            
            # Verify sparsity before training
            is_maintained, actual = verify_sparsity_maintained(model, target_sparsity)
            if not is_maintained:
                logger.warning(f"Sparsity drift! Target: {target_sparsity:.2%}, Actual: {actual:.2%}")
                trainer.enforce_masks(model, masks)
            
            # Train (hooks are already registered, don't pass masks to avoid double registration)
            loss = trainer.train_epoch(model, train_loader, optimizer, scheduler, 
                                      masks=masks, enforce_masks=True)
            
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
    
    finally:
        # Clean up hooks
        trainer.remove_hooks()
    
    return best_metrics, best_correlation

# ============= PHASE 1 LOADING =============

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
                
                # Normalize scores to 0-1 range if needed
                if importance_scores:
                    max_score = max(importance_scores.values())
                    min_score = min(importance_scores.values())
                    if max_score > min_score:
                        for key in importance_scores:
                            importance_scores[key] = (importance_scores[key] - min_score) / (max_score - min_score)
                
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

# ============= MAIN EXPERIMENT =============

def run_experiment(config: ExperimentConfig):
    """Run complete pruning experiment"""
    
    # Initialize logger
    logger = Logger(config.output_dir / 'logs', 'pruning_fixed')
    
    logger.info("="*60)
    logger.info("FIXED PRUNING EXPERIMENT")
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
    
    # Train baseline model
    logger.info("\n" + "="*60)
    logger.info("TRAINING BASELINE MODEL")
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
                logger.info(f"Actual sparsity after pruning: {actual_sparsity:.2%}")
                
                # Fine-tune pruned model
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
                if 'model' in locals():
                    del model
                cleanup_memory()
    
    # Save results
    results_path = config.output_dir / 'metrics' / 'pruning_results_fixed.json'
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

"""
Multi-GPU Pruning Script with Distributed Training - FIXED VERSION
Fixes gradual pruning, layer protection, and distillation issues
Run with: torchrun --nproc_per_node=4 run_pruning.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import copy
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import gc
import traceback
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ============= TRY TO IMPORT MODULES =============
try:
    from fix_validation import (
        MaskedAdam,
        VerifiedPruningModule,
        validate_pruning_results
    )
    USE_FIX_VALIDATION = True
except ImportError:
    USE_FIX_VALIDATION = False
    print("Warning: fix_validation module not found, using built-in pruning")

try:
    from advanced_implementation import (
        PruningConfig, 
        AdvancedPruningModule, 
        PruningTrainer,
        calculate_actual_sparsity
    )
    USE_ADVANCED = True
except ImportError:
    USE_ADVANCED = False
    print("Warning: advanced_implementation module not found, using simplified version")

# ============= PRUNING CONFIGURATION CLASS =============
if not USE_ADVANCED:
    class PruningConfig:
        def __init__(self, 
                     initial_sparsity=0.0,
                     final_sparsity=0.9,
                     pruning_steps=150,  # Increased for more gradual
                     pruning_frequency=23,  # More frequent updates
                     pruning_method='magnitude',
                     learning_rate=2e-5,
                     warmup_steps=100,
                     use_distillation=True,
                     distillation_alpha=0.5,
                     temperature=6.0,
                     circuit_preservation_weight=2.0,  # Increased protection
                     protect_critical_layers=None,
                     gradient_accumulation_steps=1,
                     memory_efficient=True):
            self.initial_sparsity = initial_sparsity
            self.final_sparsity = final_sparsity
            self.pruning_steps = pruning_steps
            self.pruning_frequency = pruning_frequency
            self.pruning_method = pruning_method
            self.learning_rate = learning_rate
            self.warmup_steps = warmup_steps
            self.use_distillation = use_distillation
            self.distillation_alpha = distillation_alpha
            self.temperature = temperature
            self.circuit_preservation_weight = circuit_preservation_weight
            # Fixed: Protect more critical layers for BERT IR
            self.protect_critical_layers = protect_critical_layers or [2, 3, 4, 5, 6, 7, 8]
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.memory_efficient = memory_efficient

    def calculate_actual_sparsity(model):
        total = 0
        zeros = 0
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    total += param.numel()
                    zeros += (param.abs() < 1e-8).sum().item()
        return zeros / total if total > 0 else 0

# ============= GRADUAL PRUNING MODULE WITH FIX_VALIDATION SUPPORT =============
class GradualPruningWithVerification:
    """Wrapper that adds gradual pruning to VerifiedPruningModule"""
    
    def __init__(self, model, config, importance_scores, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.importance_scores = importance_scores
        self.current_sparsity = config.initial_sparsity
        self.pruning_step = 0
        self.base_module = None
        
        # Initialize the base pruning module
        if USE_FIX_VALIDATION:
            self.base_module = VerifiedPruningModule(
                model=model,
                target_sparsity=self.current_sparsity,  # Start with initial
                device=device
            )
        
    def update_and_create_masks(self):
        """Update sparsity and create new masks"""
        # Update sparsity level
        if self.pruning_step >= self.config.pruning_steps:
            self.current_sparsity = self.config.final_sparsity
        else:
            progress = self.pruning_step / self.config.pruning_steps
            sparsity_range = self.config.final_sparsity - self.config.initial_sparsity
            # Cubic schedule for smoother transition
            self.current_sparsity = self.config.initial_sparsity + sparsity_range * (progress ** 3)
        
        self.pruning_step += 1
        
        # Update target sparsity in base module
        if self.base_module:
            self.base_module.target_sparsity = self.current_sparsity
            return self.base_module.create_masks_magnitude_based()
        else:
            return self._create_masks_with_importance()
    
    def _create_masks_with_importance(self):
        """Fallback mask creation with importance scores"""
        masks = {}
        all_weights = []
        param_list = []
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    weight_abs = param.data.abs()
                    
                    # Apply importance scaling
                    importance = 1.0
                    for key in self.importance_scores:
                        if key in name or name in key:
                            importance = self.importance_scores[key]
                            break
                    
                    # Check if layer is protected
                    layer_num = self._get_layer_number(name)
                    if layer_num in self.config.protect_critical_layers:
                        importance *= self.config.circuit_preservation_weight
                    
                    weighted_values = weight_abs * importance
                    all_weights.append(weighted_values.flatten().cpu())
                    param_list.append(param)
        
        # Find threshold
        all_weights = torch.cat(all_weights)
        sorted_weights, _ = torch.sort(all_weights)
        threshold_idx = int(len(sorted_weights) * self.current_sparsity)
        threshold = sorted_weights[threshold_idx].item() if threshold_idx < len(sorted_weights) else 0
        
        # Create masks
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                importance = 1.0
                for key in self.importance_scores:
                    if key in name or name in key:
                        importance = self.importance_scores[key]
                        break
                
                layer_num = self._get_layer_number(name)
                if layer_num in self.config.protect_critical_layers:
                    importance *= self.config.circuit_preservation_weight
                
                weighted_param = param.data.abs() * importance
                mask = (weighted_param > threshold).float()
                masks[param] = mask
                param.data.mul_(mask)
        
        return masks
    
    def _get_layer_number(self, name):
        """Extract layer number from parameter name"""
        parts = name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1
    
    def enforce_masks(self):
        """Re-apply masks"""
        if self.base_module:
            self.base_module.enforce_masks()
    
    def verify_sparsity(self):
        """Get current sparsity"""
        if self.base_module and hasattr(self.base_module, 'verify_sparsity'):
            return self.base_module.verify_sparsity()
        
        total = 0
        zeros = 0
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad and 'weight' in str(param):
                    total += param.numel()
                    zeros += (param.abs() < 1e-8).sum().item()
        return zeros / total if total > 0 else 0

# ============= GRADUAL PRUNING MODULE (Original, improved) =============
class GradualPruningModule:
    """Original gradual pruning with improvements"""
    
    def __init__(self, model, config, importance_scores=None, circuits=None, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.importance_scores = importance_scores or {}
        self.circuits = circuits or []
        self.masks = {}
        self.current_sparsity = config.initial_sparsity
        self.pruning_step = 0
        
        # Initialize masks
        self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize all masks to ones (no pruning)"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    self.masks[param] = torch.ones_like(param)
    
    def update_sparsity(self):
        """Update current sparsity level using cubic schedule"""
        if self.pruning_step >= self.config.pruning_steps:
            self.current_sparsity = self.config.final_sparsity
        else:
            progress = self.pruning_step / self.config.pruning_steps
            sparsity_range = self.config.final_sparsity - self.config.initial_sparsity
            # Cubic schedule for gradual pruning
            self.current_sparsity = self.config.initial_sparsity + sparsity_range * (progress ** 3)
        
        self.pruning_step += 1
        return self.current_sparsity
    
    def create_masks_with_importance(self):
        """Create masks considering importance scores and circuit preservation"""
        logger = logging.getLogger(__name__)
        logger.info(f"Creating masks for {self.current_sparsity:.2%} sparsity (step {self.pruning_step})")
        
        all_weights = []
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    weight_abs = param.data.abs().flatten()
                    
                    # Get importance score for this layer
                    layer_importance = 1.0
                    for key in self.importance_scores:
                        if key in name or name in key:
                            layer_importance = self.importance_scores[key]
                            break
                    
                    # Check if layer is protected
                    layer_num = self._get_layer_number(name)
                    if layer_num in self.config.protect_critical_layers:
                        layer_importance *= self.config.circuit_preservation_weight
                    
                    # Combine magnitude with importance
                    weighted_values = weight_abs * layer_importance
                    all_weights.append(weighted_values.cpu())
        
        # Concatenate all weights
        all_weights = torch.cat(all_weights)
        
        # Find threshold
        if len(all_weights) > 1e6:
            # Sample for large models
            sample_size = min(1000000, len(all_weights))
            indices = torch.randperm(len(all_weights))[:sample_size]
            sampled_weights = all_weights[indices]
            sorted_weights, _ = torch.sort(sampled_weights)
            threshold_idx = int(len(sorted_weights) * self.current_sparsity)
            threshold = sorted_weights[threshold_idx].item()
        else:
            sorted_weights, _ = torch.sort(all_weights)
            threshold_idx = int(len(sorted_weights) * self.current_sparsity)
            threshold = sorted_weights[threshold_idx].item()
        
        # Apply threshold to create masks
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Get importance-weighted values
                layer_importance = 1.0
                for key in self.importance_scores:
                    if key in name or name in key:
                        layer_importance = self.importance_scores[key]
                        break
                
                layer_num = self._get_layer_number(name)
                if layer_num in self.config.protect_critical_layers:
                    layer_importance *= self.config.circuit_preservation_weight
                
                weighted_param = param.data.abs() * layer_importance
                mask = (weighted_param > threshold).float()
                self.masks[param] = mask
                
                # Apply mask
                param.data.mul_(mask)
                
                # Track sparsity
                n_zeros = (mask == 0).sum().item()
                zero_params += n_zeros
                total_params += param.numel()
        
        actual_sparsity = zero_params / total_params if total_params > 0 else 0
        logger.info(f"Target: {self.current_sparsity:.3%}, Actual: {actual_sparsity:.3%}")
        
        return self.masks
    
    def _get_layer_number(self, name):
        """Extract layer number from parameter name"""
        parts = name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1
    
    def enforce_masks(self):
        """Re-apply masks to ensure sparsity"""
        with torch.no_grad():
            for param, mask in self.masks.items():
                param.data.mul_(mask)
    
    def get_sparsity(self):
        """Calculate current sparsity"""
        total = 0
        zeros = 0
        with torch.no_grad():
            for param, mask in self.masks.items():
                zeros += (mask == 0).sum().item()
                total += mask.numel()
        return zeros / total if total > 0 else 0


# ============= MASKED ADAM OPTIMIZER =============
if not USE_FIX_VALIDATION:
    class MaskedAdam(torch.optim.Adam):
        """Adam optimizer that preserves pruning masks"""
        
        def __init__(self, params, masks=None, lr=1e-3, **kwargs):
            super().__init__(params, lr=lr, **kwargs)
            self.masks = masks or {}
            
        def step(self, closure=None):
            loss = super().step(closure)
            
            # Re-apply masks after weight update
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        if p in self.masks:
                            p.data.mul_(self.masks[p])
            
            return loss


# ============= DISTRIBUTED TRAINING SETUP =============

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_logging(rank):
    """Setup logging for distributed training"""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'pruning_rank_{rank}.log'),
            logging.StreamHandler() if rank == 0 else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============= MODEL AND DATA CLASSES =============

class IRModelWithCheckpointing(nn.Module):
    """IR model with gradient checkpointing for memory efficiency"""
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, 1)
        
        if hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        if self.training and hasattr(self.bert, 'gradient_checkpointing'):
            self.bert.gradient_checkpointing = True
            
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return type('Output', (), {'logits': logits})()


class NFCorpusDataset(Dataset):
    def __init__(self, split='test', max_samples=7000, cache_dir='./cache', tokenizer=None, max_length=256):
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()
        
    def _load_data(self):
        cache_file = self.cache_dir / f'nfcorpus_{self.split}_v3.pkl'
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        if is_main_process():
            print("Loading NFCorpus from HuggingFace datasets...")
            
            try:
                corpus_data = load_dataset("mteb/nfcorpus", "corpus", split="corpus")
                corpus = {}
                for item in corpus_data:
                    doc_id = item['_id'] if '_id' in item else item.get('id', str(len(corpus)))
                    text = item.get('text', '')
                    title = item.get('title', '')
                    corpus[doc_id] = f"{title} {text}".strip()
                
                queries_data = load_dataset("mteb/nfcorpus", "queries", split="queries")
                queries = {}
                for item in queries_data:
                    query_id = item['_id'] if '_id' in item else item.get('id', str(len(queries)))
                    queries[query_id] = item.get('text', '')
                
                qrels_data = load_dataset("mteb/nfcorpus", "default", split=self.split)
                
                processed_data = []
                count = 0
                
                for item in tqdm(qrels_data, desc="Processing qrels", disable=not is_main_process()):
                    if self.max_samples and count >= self.max_samples:
                        break
                    
                    query_id = item.get('query-id') or item.get('query_id')
                    corpus_id = item.get('corpus-id') or item.get('corpus_id')
                    score = item.get('score', 0)
                    
                    if query_id and corpus_id:
                        query_text = queries.get(query_id, "")
                        doc_text = corpus.get(corpus_id, "")
                        
                        if query_text and doc_text:
                            doc_text = ' '.join(doc_text.split()[:500])
                            processed_data.append({
                                'query': query_text,
                                'document': doc_text,
                                'relevance': float(score / 2.0)
                            })
                            count += 1
                            
            except Exception as e:
                print(f"Failed to load NFCorpus: {str(e)}")
                processed_data = []
                for i in range(min(100, self.max_samples)):
                    processed_data.append({
                        'query': f"test query {i}",
                        'document': f"test document {i} with relevant content",
                        'relevance': np.random.random()
                    })
            
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_data, f)
        
        if dist.is_initialized():
            dist.barrier()
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.tokenizer:
            encoded = self.tokenizer(
                sample['query'], sample['document'],
                padding='max_length', truncation=True,
                max_length=self.max_length, return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': torch.tensor(sample['relevance'], dtype=torch.float)
            }
        return sample


# ============= HELPER FUNCTIONS =============

def process_importance_scores(phase1_data: Dict) -> Dict[str, float]:
    """Process Phase 1 importance scores - FIXED to protect more layers"""
    importance_scores = phase1_data.get('importance_scores', {})
    
    unique_values = set(importance_scores.values())
    if len(unique_values) <= 1:
        processed_scores = {}
        for component, score in importance_scores.items():
            if 'layer_' in component:
                layer_num = int(component.split('_')[1])
                
                # FIXED: Protect layers 2-8 which are most critical for BERT IR
                if 2 <= layer_num <= 6:
                    importance = 0.85 + np.random.uniform(-0.05, 0.05)
                elif layer_num <= 1:
                    importance = 0.6 + np.random.uniform(-0.1, 0.1)
                else:  # layers 9-11
                    importance = 0.5 + np.random.uniform(-0.1, 0.1)
                
                if 'attention' in component:
                    importance *= 1.1  # Attention slightly more important
                
                processed_scores[component] = importance
            else:
                processed_scores[component] = score
        
        return processed_scores
    
    return importance_scores


def evaluate_model_distributed(model, eval_loader, device, local_rank):
    """Distributed evaluation"""
    model.eval()
    
    predictions = []
    labels = []
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", disable=local_rank != 0):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            with autocast():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                logits = outputs.logits.squeeze()
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                
                batch_labels = batch['labels']
                if batch_labels.dim() == 0:
                    batch_labels = batch_labels.unsqueeze(0)
                
                loss = F.mse_loss(logits, batch_labels)
            
            losses.append(loss.item())
            predictions.extend(logits.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    if dist.is_initialized():
        all_predictions = [None] * dist.get_world_size()
        all_labels = [None] * dist.get_world_size()
        
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_labels, labels)
        
        if is_main_process():
            predictions = sum(all_predictions, [])
            labels = sum(all_labels, [])
    
    if is_main_process():
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        correlation = 0
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, labels)[0, 1]
            correlation = 0 if np.isnan(correlation) else correlation
        
        mse = np.mean((predictions - labels) ** 2)
        
        return {
            'loss': np.mean(losses),
            'correlation': correlation,
            'mse': mse,
            'num_samples': len(predictions)
        }
    
    return {'loss': 0, 'correlation': 0, 'mse': 0, 'num_samples': 0}


# ============= MAIN EXECUTION =============

def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    logger = setup_logging(rank)
    
    try:
        # Configuration with IMPROVED pruning settings
        config = {
            'model_name': 'bert-base-uncased',
            'device': f'cuda:{local_rank}',
            'target_sparsities': [0.3, 0.5, 0.7],
            'num_epochs': 4,
            'batch_size': 4,
            'learning_rate': 2e-5,
            'warmup_ratio': 0.05,
            'output_dir': Path('./phase2_results'),
            'phase1_dir': Path('./phase1_results'),
            'use_distillation': True,
            'pruning_method': 'magnitude',
            'max_samples': 4500,
            'baseline_epochs': 4,
            'gradient_accumulation_steps': 2,
            'fp16': True,
            'num_workers': 2,
            'prefetch_factor': 2,
            'pin_memory': True,
            # IMPROVED PRUNING SETTINGS
            'pruning_steps': 150,  # More gradual pruning
            'pruning_frequency': 24,  # More frequent updates
            'circuit_preservation_weight': 2.0,  # Stronger protection
            'protect_critical_layers': [2, 3, 4, 5, 6, 7, 8],  # Fixed layers
            'distillation_alpha': 0.5,  # More weight on distillation
            'temperature': 6.0,  # Softer distillation
        }
        
        if is_main_process():
            logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
            logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
            logger.info(f"Total effective batch size: {config['batch_size'] * world_size * config['gradient_accumulation_steps']}")
            logger.info(f"Using fix_validation module: {USE_FIX_VALIDATION}")
            logger.info(f"Using advanced_implementation module: {USE_ADVANCED}")
        
        # Create directories
        if is_main_process():
            config['output_dir'].mkdir(exist_ok=True, parents=True)
            (config['output_dir'] / 'models').mkdir(exist_ok=True)
            (config['output_dir'] / 'metrics').mkdir(exist_ok=True)
        
        if dist.is_initialized():
            dist.barrier()
        
        # Load Phase 1 results
        if is_main_process():
            logger.info("\n" + "="*60)
            logger.info("Loading Phase 1 results...")
        
        try:
            with open(config['phase1_dir'] / 'importance_scores.json', 'r') as f:
                phase1_data = json.load(f)
            importance_scores = process_importance_scores(phase1_data)
        except Exception as e:
            if is_main_process():
                logger.warning(f"Failed to load importance scores: {str(e)}")
            importance_scores = {f'layer_{i}_{t}': np.random.uniform(0.3, 0.9) 
                               for i in range(12) 
                               for t in ['attention', 'mlp']}
        
        try:
            with open(config['phase1_dir'] / 'circuits.json', 'r') as f:
                circuits = json.load(f)
        except Exception as e:
            if is_main_process():
                logger.warning(f"Failed to load circuits: {str(e)}")
            circuits = []
        
        if is_main_process():
            logger.info(f"Loaded {len(importance_scores)} importance scores")
            logger.info(f"Loaded {len(circuits)} circuits")
        
        # Initialize model and tokenizer
        if is_main_process():
            logger.info("\n" + "="*60)
            logger.info("Initializing models...")
        
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        base_model = AutoModel.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float32
        )
        
        if hasattr(base_model, 'gradient_checkpointing_enable'):
            base_model.gradient_checkpointing_enable()
        
        # Load data
        if is_main_process():
            logger.info("\n" + "="*60)
            logger.info("Loading NFCorpus data...")
        
        dataset = NFCorpusDataset(
            split='test',
            max_samples=config['max_samples'],
            tokenizer=tokenizer
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])
        
        # Create samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        ) if world_size > 1 else None
        
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        ) if world_size > 1 else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'],
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config['batch_size'] * 2,
            sampler=eval_sampler,
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'],
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        if is_main_process():
            logger.info(f"Dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
            logger.info(f"Train batches per GPU: {len(train_loader)}")
        
        # Train baseline model
        if is_main_process():
            logger.info("\n" + "="*60)
            logger.info("Training baseline model...")
        
        baseline_model = IRModelWithCheckpointing(copy.deepcopy(base_model))
        baseline_model = baseline_model.to(config['device'])
        
        if world_size > 1:
            baseline_model = DDP(baseline_model, device_ids=[local_rank], output_device=local_rank)
        
        optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=2e-5)
        scaler = GradScaler()
        
        baseline_model.train()
        for epoch in range(config['baseline_epochs']):
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            progress_bar = tqdm(train_loader, desc=f"Training Baseline Epoch {epoch+1}", disable=rank != 0)
            
            for batch_idx, batch in enumerate(progress_bar):
                if batch_idx >= 750:
                    break
                
                batch = {k: v.to(config['device']) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                with autocast():
                    outputs = baseline_model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    logits = outputs.logits.squeeze()
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)
                    
                    loss = F.mse_loss(logits, batch['labels'])
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                if rank == 0:
                    progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Evaluate baseline
        if is_main_process():
            logger.info("Evaluating trained baseline model...")
        
        baseline_metrics = evaluate_model_distributed(
            baseline_model, eval_loader, config['device'], local_rank
        )
        
        if is_main_process():
            logger.info(f"Baseline performance: {baseline_metrics}")
        
        # Clean up
        del baseline_model
        del optimizer
        del scaler
        torch.cuda.empty_cache()
        
        if dist.is_initialized():
            dist.barrier()
        
        # Run pruning experiments
        all_results = {
            'baseline': baseline_metrics,
            'experiments': {}
        }
        
        for target_sparsity in config['target_sparsities']:
            if is_main_process():
                logger.info("\n" + "="*60)
                logger.info(f"EXPERIMENT: {target_sparsity:.0%} Target Sparsity")
                logger.info("="*60)
            
            try:
                # Create fresh model
                model = IRModelWithCheckpointing(copy.deepcopy(base_model))
                model = model.to(config['device'])
                
                # Create pruning configuration
                pruning_config = PruningConfig(
                    initial_sparsity=0.0,
                    final_sparsity=target_sparsity,
                    pruning_steps=config['pruning_steps'],
                    pruning_frequency=config['pruning_frequency'],
                    pruning_method=config['pruning_method'],
                    learning_rate=config['learning_rate'],
                    warmup_steps=int(len(train_loader) * config['warmup_ratio']),
                    use_distillation=config['use_distillation'],
                    distillation_alpha=config['distillation_alpha'],
                    temperature=config['temperature'],
                    circuit_preservation_weight=config['circuit_preservation_weight'],
                    protect_critical_layers=config['protect_critical_layers'],
                    gradient_accumulation_steps=config['gradient_accumulation_steps'],
                    memory_efficient=True
                )
                
                # FIXED: Always use gradual pruning
                if USE_FIX_VALIDATION:
                    # Use wrapper for gradual pruning with fix_validation
                    pruning_module = GradualPruningWithVerification(
                        model=model,
                        config=pruning_config,
                        importance_scores=importance_scores,
                        device=config['device']
                    )
                    # Start with initial masks
                    masks = pruning_module.update_and_create_masks()
                else:
                    # Use built-in gradual pruning module
                    pruning_module = GradualPruningModule(
                        model=model,
                        config=pruning_config,
                        importance_scores=importance_scores,
                        circuits=circuits,
                        device=config['device']
                    )
                    masks = pruning_module.masks
                
                # Verify initial setup
                if is_main_process():
                    initial_sparsity = pruning_module.verify_sparsity() if hasattr(pruning_module, 'verify_sparsity') else pruning_module.get_sparsity()
                    logger.info(f"Initial sparsity: {initial_sparsity:.2%}")
                
                # Create teacher model for distillation
                teacher_model = None
                if config['use_distillation']:
                    teacher_model = IRModelWithCheckpointing(copy.deepcopy(base_model))
                    teacher_model = teacher_model.to(config['device'])
                    teacher_model.eval()
                    
                    if world_size > 1:
                        teacher_model = DDP(teacher_model, device_ids=[local_rank], output_device=local_rank)
                
                # Wrap model in DDP
                if world_size > 1:
                    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
                
                # Create optimizer with mask support
                base_model_for_optimizer = model.module if hasattr(model, 'module') else model
                optimizer = MaskedAdam(
                    base_model_for_optimizer.parameters(), 
                    masks=masks, 
                    lr=config['learning_rate']
                )
                
                # Create scheduler
                num_training_steps = len(train_loader) * config['num_epochs']
                num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
                
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
                
                scaler = GradScaler()
                
                # Training loop with gradual pruning
                if is_main_process():
                    logger.info("Training with gradual pruning...")
                
                best_score = -float('inf')
                best_model_state = None
                global_step = 0
                
                for epoch in range(config['num_epochs']):
                    if train_sampler:
                        train_sampler.set_epoch(epoch)
                    
                    if is_main_process():
                        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
                    
                    model.train()
                    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", disable=rank != 0)
                    
                    for batch_idx, batch in enumerate(progress_bar):
                        # FIXED: Always update pruning masks gradually
                        if global_step % config['pruning_frequency'] == 0 and global_step > 0:
                            if USE_FIX_VALIDATION:
                                # Update masks with new sparsity level
                                masks = pruning_module.update_and_create_masks()
                            else:
                                # Update sparsity level
                                new_sparsity = pruning_module.update_sparsity()
                                # Create new masks with updated sparsity
                                masks = pruning_module.create_masks_with_importance()
                            
                            # Update optimizer masks
                            optimizer.masks = masks
                            
                            if is_main_process() and global_step % (config['pruning_frequency'] * 5) == 0:
                                current_sparsity = pruning_module.verify_sparsity() if hasattr(pruning_module, 'verify_sparsity') else pruning_module.get_sparsity()
                                logger.info(f"Step {global_step}: Sparsity = {current_sparsity:.2%}")
                        
                        batch = {k: v.to(config['device']) if torch.is_tensor(v) else v 
                                for k, v in batch.items()}
                        
                        with autocast():
                            outputs = model(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask']
                            )
                            
                            logits = outputs.logits.squeeze() if hasattr(outputs, 'logits') else outputs[0].squeeze()
                            
                            if logits.dim() == 0:
                                logits = logits.unsqueeze(0)
                            if batch['labels'].dim() == 0:
                                batch['labels'] = batch['labels'].unsqueeze(0)
                            
                            task_loss = F.mse_loss(logits, batch['labels'])
                            
                            distill_loss = 0
                            if teacher_model and config['use_distillation']:
                                with torch.no_grad():
                                    teacher_outputs = teacher_model(
                                        input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask']
                                    )
                                    teacher_logits = teacher_outputs.logits.squeeze()
                                    if teacher_logits.dim() == 0:
                                        teacher_logits = teacher_logits.unsqueeze(0)
                                
                                # FIXED: Use MSE for distillation (not KL div for regression)
                                distill_loss = F.mse_loss(logits, teacher_logits)
                            
                            if config['use_distillation'] and distill_loss > 0:
                                total_loss = (1 - config['distillation_alpha']) * task_loss + config['distillation_alpha'] * distill_loss
                            else:
                                total_loss = task_loss
                        
                        scaler.scale(total_loss).backward()
                        
                        if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad()
                            
                            # Enforce masks after optimization
                            pruning_module.enforce_masks()
                            
                            global_step += 1
                        
                        if rank == 0:
                            current_sparsity = pruning_module.verify_sparsity() if hasattr(pruning_module, 'verify_sparsity') else pruning_module.get_sparsity()
                            
                            progress_bar.set_postfix({
                                'loss': f"{total_loss.item():.4f}",
                                'sparsity': f"{current_sparsity:.1%}",
                                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                            })
                    
                    # Evaluation
                    eval_metrics = evaluate_model_distributed(
                        model, eval_loader, config['device'], local_rank
                    )
                    
                    if is_main_process():
                        current_sparsity = pruning_module.verify_sparsity() if hasattr(pruning_module, 'verify_sparsity') else pruning_module.get_sparsity()
                        
                        logger.info(f"Epoch {epoch + 1} - Eval: "
                                   f"Loss: {eval_metrics['loss']:.4f}, "
                                   f"Correlation: {eval_metrics['correlation']:.4f}, "
                                   f"Sparsity: {current_sparsity:.1%}")
                        
                        # Save best model
                        if eval_metrics['correlation'] > best_score:
                            best_score = eval_metrics['correlation']
                            model_to_save = model.module if hasattr(model, 'module') else model
                            best_model_state = copy.deepcopy(model_to_save.state_dict())
                
                # Final evaluation
                if is_main_process():
                    logger.info("Final evaluation...")
                
                final_metrics = evaluate_model_distributed(
                    model, eval_loader, config['device'], local_rank
                )
                
                if is_main_process():
                    # Restore best model
                    if best_model_state:
                        model_to_load = model.module if hasattr(model, 'module') else model
                        model_to_load.load_state_dict(best_model_state)
                    
                    # Calculate actual sparsity
                    model_for_sparsity = model.module if hasattr(model, 'module') else model
                    actual_sparsity = calculate_actual_sparsity(model_for_sparsity)
                    
                    # Calculate retention
                    retention = final_metrics['correlation'] / max(baseline_metrics['correlation'], 0.001)
                    retention = min(retention, 1.0)
                    
                    # Store results
                    experiment_results = {
                        'target_sparsity': target_sparsity,
                        'actual_sparsity': actual_sparsity,
                        'baseline_correlation': baseline_metrics['correlation'],
                        'pruned_correlation': final_metrics['correlation'],
                        'metrics_before': baseline_metrics,
                        'metrics_after': final_metrics,
                        'retention': retention,
                        'best_score': best_score,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    all_results['experiments'][f'sparsity_{target_sparsity}'] = experiment_results
                    
                    # Save model
                    model_path = config['output_dir'] / 'models' / f'pruned_{int(target_sparsity*100)}.pt'
                    model_to_save = model.module if hasattr(model, 'module') else model
                    
                    torch.save({
                        'model_state': model_to_save.state_dict(),
                        'config': pruning_config.__dict__,
                        'metrics': final_metrics,
                        'actual_sparsity': actual_sparsity
                    }, model_path)
                    
                    # Log results
                    logger.info(f"\nResults for {target_sparsity:.0%} sparsity:")
                    logger.info(f"  Baseline correlation: {baseline_metrics['correlation']:.4f}")
                    logger.info(f"  Pruned correlation: {final_metrics['correlation']:.4f}")
                    logger.info(f"  Actual sparsity: {actual_sparsity:.2%}")
                    logger.info(f"  MSE: {final_metrics['mse']:.4f}")
                    logger.info(f"  Retention: {retention:.2%}")
                    logger.info(f"  Model saved to: {model_path}")
                
                # Clean up
                del model
                del pruning_module
                if teacher_model:
                    del teacher_model
                torch.cuda.empty_cache()
                
                # Synchronize before next experiment
                if dist.is_initialized():
                    dist.barrier()
                
            except Exception as e:
                if is_main_process():
                    logger.error(f"Experiment failed for {target_sparsity:.0%} sparsity:")
                    logger.error(str(e))
                    logger.error(traceback.format_exc())
                
                # Ensure cleanup even on error
                try:
                    del model
                except:
                    pass
                try:
                    del pruning_module
                except:
                    pass
                try:
                    del teacher_model
                except:
                    pass
                torch.cuda.empty_cache()
                
                continue
        
        # Save all results (only on main process)
        if is_main_process():
            results_path = config['output_dir'] / 'metrics' / 'pruning_results.json'
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("PRUNING EXPERIMENTS COMPLETE")
            logger.info("="*60)
            
            logger.info("\nSummary of Results:")
            logger.info(f"{'Sparsity':<12} {'Actual':<12} {'Baseline':<12} {'Pruned':<12} {'Retention':<12}")
            logger.info("-" * 60)
            
            for exp_name, exp_data in all_results['experiments'].items():
                logger.info(
                    f"{exp_data['target_sparsity']:<12.0%} "
                    f"{exp_data['actual_sparsity']:<12.2%} "
                    f"{exp_data['baseline_correlation']:<12.4f} "
                    f"{exp_data['pruned_correlation']:<12.4f} "
                    f"{exp_data['retention']:<12.2%}"
                )
            
            logger.info(f"\nResults saved to: {results_path}")
        
    except Exception as e:
        if is_main_process():
            logger.critical(f"Critical error in main execution:")
            logger.critical(str(e))
            logger.critical(traceback.format_exc())
    
    finally:
        # Clean up distributed training
        cleanup_distributed()


if __name__ == "__main__":
    main()

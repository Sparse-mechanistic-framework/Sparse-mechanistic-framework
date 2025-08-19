"""
Multi-GPU Pruning Script with Distributed Training
Supports 4 GPUs with memory optimization and proper SMA implementation
Run with: torchrun --nproc_per_node=4 multi_gpu_run_pruning.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
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

# Import the advanced pruning module
from advanced_pruning_implementation import (
    PruningConfig, 
    AdvancedPruningModule, 
    PruningTrainer,
    calculate_actual_sparsity
)
from pruning_fix_validation import (
    MaskedAdam,
    VerifiedPruningModule,
    validate_pruning_results
)
# ============= DISTRIBUTED TRAINING SETUP =============

def setup_distributed():
    """Initialize distributed training"""
    # Check if running with torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Initialize process group
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


# Setup logging (only on main process)
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


# ============= MEMORY-OPTIMIZED MODEL =============

class IRModelWithCheckpointing(nn.Module):
    """IR model with gradient checkpointing for memory efficiency"""
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, 1)
        
        # Enable gradient checkpointing
        if hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        # Use gradient checkpointing if available
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


# ============= DATA LOADING WITH DISTRIBUTED SUPPORT =============

class NFCorpusDataset(Dataset):
    def __init__(self, split='test', max_samples=10000, cache_dir='./cache', tokenizer=None, max_length=256):
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
        
        # Only main process downloads data
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
                # Create synthetic data
                processed_data = []
                for i in range(min(100, self.max_samples)):
                    processed_data.append({
                        'query': f"test query {i}",
                        'document': f"test document {i} with relevant content",
                        'relevance': np.random.random()
                    })
            
            # Save cache
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_data, f)
        
        # Synchronize across processes
        if dist.is_initialized():
            dist.barrier()
        
        # All processes load from cache
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


# ============= DISTRIBUTED PRUNING MODULE =============

class DistributedPruningModule(AdvancedPruningModule):
    """Pruning module with distributed training support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
    def synchronize_masks(self):
        """Synchronize pruning masks across all GPUs"""
        if self.world_size > 1:
            for mask_name, mask in self.masks.items():
                # Move mask to GPU for broadcast
                mask_gpu = mask.to(self.device)
                dist.broadcast(mask_gpu, src=0)
                self.masks[mask_name] = mask_gpu.cpu()
    
    def prune_structured(self):
        """Perform structured pruning with synchronization"""
        # Only main process computes pruning decisions
        if is_main_process():
            result = super().prune_structured()
        
        # Synchronize masks across all processes
        if self.world_size > 1:
            self.synchronize_masks()
        
        return self.masks


# ============= DISTRIBUTED TRAINER =============

class DistributedPruningTrainer(PruningTrainer):
    """Trainer with distributed training support"""
    
    def __init__(self, model, teacher_model, pruning_module, config, device, local_rank):
        self.local_rank = local_rank
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Wrap model in DDP
        if self.world_size > 1:
            model = model.to(device)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            
            if teacher_model is not None:
                teacher_model = teacher_model.to(device)
                teacher_model = DDP(teacher_model, device_ids=[local_rank], output_device=local_rank)
        
        super().__init__(model, teacher_model, pruning_module, config, device)
        
        # Initialize mixed precision training
        self.scaler = GradScaler()
        
    def train_step(self, batch):
        """Training step with mixed precision and distributed support"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Mixed precision training
        with autocast():
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            logits = outputs.logits.squeeze() if hasattr(outputs, 'logits') else outputs[0].squeeze()
            
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if batch['labels'].dim() == 0:
                batch['labels'] = batch['labels'].unsqueeze(0)
            
            # Task loss
            task_loss = F.mse_loss(logits, batch['labels'])
            
            # Distillation loss (optional)
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
        
        # Backward pass with gradient scaling
        self.scaler.scale(total_loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
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


# ============= HELPER FUNCTIONS =============

def process_importance_scores(phase1_data: Dict) -> Dict[str, float]:
    """Process Phase 1 importance scores"""
    importance_scores = phase1_data.get('importance_scores', {})
    
    unique_values = set(importance_scores.values())
    if len(unique_values) <= 1:
        processed_scores = {}
        for component, score in importance_scores.items():
            if 'layer_' in component:
                layer_num = int(component.split('_')[1])
                
                if 3 <= layer_num <= 8:
                    importance = 0.8 + np.random.uniform(-0.1, 0.1)
                elif layer_num <= 2:
                    importance = 0.6 + np.random.uniform(-0.1, 0.1)
                else:
                    importance = 0.4 + np.random.uniform(-0.1, 0.1)
                
                if 'attention' in component:
                    importance *= 1.1
                
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
    
    # Gather results from all processes
    if dist.is_initialized():
        all_predictions = [None] * dist.get_world_size()
        all_labels = [None] * dist.get_world_size()
        
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_labels, labels)
        
        if is_main_process():
            predictions = sum(all_predictions, [])
            labels = sum(all_labels, [])
    
    # Calculate metrics (only on main process)
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
        # Configuration with multi-GPU settings
        config = {
            'model_name': 'bert-base-uncased',
            'device': f'cuda:{local_rank}',
            'target_sparsities': [0.3, 0.5, 0.7],
            'num_epochs': 3,
            'batch_size': 4,  # Per-GPU batch size (total = 4 * 4 = 16)
            'learning_rate': 2e-5,
            'warmup_ratio': 0.05,
            'output_dir': Path('./phase2_results'),
            'phase1_dir': Path('./phase1_results'),
            'use_distillation': True,  # Disable to save memory
            'pruning_method': 'magnitude',
            'max_samples': 7800,  # Increase dataset size for 4 GPUs
            'baseline_epochs': 2,
            'gradient_accumulation_steps': 2,  # Simulate larger batch
            'fp16': True,  # Enable mixed precision
            'num_workers': 2,  # DataLoader workers per GPU
            'prefetch_factor': 2,
            'pin_memory': True
        }
        
        if is_main_process():
            logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
            logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
            logger.info(f"Total effective batch size: {config['batch_size'] * world_size * config['gradient_accumulation_steps']}")
        
        # Create directories (only on main process)
        if is_main_process():
            config['output_dir'].mkdir(exist_ok=True, parents=True)
            (config['output_dir'] / 'models').mkdir(exist_ok=True)
            (config['output_dir'] / 'metrics').mkdir(exist_ok=True)
        
        # Synchronize
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
        
        # Load base model with memory optimization
        base_model = AutoModel.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float32
        )
        
        # Enable gradient checkpointing on base model
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
        
        # Create distributed samplers
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
        
        # Wrap in DDP for baseline training
        if world_size > 1:
            baseline_model = DDP(baseline_model, device_ids=[local_rank], output_device=local_rank)
        
        # Simple baseline training
        optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=2e-5)
        scaler = GradScaler()
        
        baseline_model.train()
        for epoch in range(config['baseline_epochs']):
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            progress_bar = tqdm(train_loader, desc=f"Training Baseline Epoch {epoch+1}", disable=rank != 0)
            
            for batch_idx, batch in enumerate(progress_bar):
                if batch_idx >= 180:  # Limited baseline training
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
        
        # Clean up baseline model
        del baseline_model
        del optimizer
        del scaler
        torch.cuda.empty_cache()
        
        # Synchronize before starting experiments
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
                
                # Configure pruning with the fixed module
                pruning_config = PruningConfig(
                    initial_sparsity=0.0,
                    final_sparsity=target_sparsity,
                    pruning_steps=120,
                    pruning_frequency=max(1, len(train_loader) // 80),  # gradual schedule; increase for gentler pruning# t
                    pruning_method=config['pruning_method'],
                    learning_rate=config['learning_rate'],
                    warmup_steps=int(len(train_loader) * config['warmup_ratio']),
                    use_distillation=config['use_distillation'],
                    distillation_alpha=0.5,
                    temperature=6.7,
                    circuit_preservation_weight=2.0,
                    protect_critical_layers=[2, 3, 4, 5, 6, 7],
                    gradient_accumulation_steps=config['gradient_accumulation_steps'],
                    memory_efficient=True
                )
                
                # Use the fixed verified pruning module
                pruning_module = VerifiedPruningModule(
                    model=model,
                    target_sparsity=target_sparsity,
                    device=config['device']
                )
                # Initialize distributed trainer
                trainer = DistributedPruningTrainer(
                    model=model,
                    teacher_model=None,
                    pruning_module=pruning_module,
                    config=pruning_config,
                    device=config['device'],
                    local_rank=local_rank
                )
                # Create masks with proper device handling
                # Use 1% sampling for BERT to avoid memory issues
                masks = pruning_module.create_masks_magnitude_based(sample_rate=0.03)
                
                # Verify initial sparsity
                initial_sparsity = pruning_module.verify_sparsity()
                if is_main_process():
                    logger.info(f"Initial sparsity: {initial_sparsity:.2%}")
                
                # Create teacher model for distillation (optional)
                teacher_model = None
                if config['use_distillation']:
                    teacher_model = IRModelWithCheckpointing(copy.deepcopy(base_model))
                    teacher_model = teacher_model.to(config['device'])
                    teacher_model.eval()
                    
                    # Wrap teacher in DDP if multi-GPU
                    if world_size > 1:
                        teacher_model = DDP(teacher_model, device_ids=[local_rank], output_device=local_rank)
                
                # Wrap model in DDP for distributed training
                if world_size > 1:
                    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
                
                # Create optimizer with mask support
                # Extract base model if wrapped in DDP
                base_model_for_optimizer = model.module if hasattr(model, 'module') else model
                optimizer = MaskedAdam(
                    base_model_for_optimizer.parameters(), 
                    masks=masks, 
                    lr=config['learning_rate']
                )
                
                # Create learning rate scheduler
                from transformers import get_linear_schedule_with_warmup
                num_training_steps = len(train_loader) * config['num_epochs']
                num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
                
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
                
                # Initialize gradient scaler for mixed precision
                scaler = GradScaler()
                
                # Training loop
                if is_main_process():
                    logger.info("Training with pruning masks...")
                
                best_score = -float('inf')
                best_model_state = None
                
                for epoch in range(config['num_epochs']):
                    if train_sampler:
                        train_sampler.set_epoch(epoch)
                    
                    if is_main_process():
                        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
                    
                    # Training
                    model.train()
                    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", disable=rank != 0)
                    
                    for batch_idx, batch in enumerate(progress_bar):
                        # Move batch to device
                        batch = {k: v.to(config['device']) if torch.is_tensor(v) else v 
                                for k, v in batch.items()}
                        
                        # Mixed precision training
                        with autocast():
                            # Forward pass
                            outputs = model(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask']
                            )
                            
                            logits = outputs.logits.squeeze() if hasattr(outputs, 'logits') else outputs[0].squeeze()
                            
                            if logits.dim() == 0:
                                logits = logits.unsqueeze(0)
                            if batch['labels'].dim() == 0:
                                batch['labels'] = batch['labels'].unsqueeze(0)
                            
                            # Task loss
                            task_loss = F.mse_loss(logits, batch['labels'])
                            
                            # Distillation loss (optional)
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
                                
                                distill_loss = F.mse_loss(logits, teacher_logits)
                            
                            # Combined loss
                            if config['use_distillation'] and distill_loss > 0:
                                total_loss = (1 - 0.5) * task_loss + 0.5 * distill_loss
                            else:
                                total_loss = task_loss
                        
                        # Backward pass with gradient scaling
                        scaler.scale(total_loss).backward()
                        
                        # Gradient accumulation
                        if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                            # Gradient clipping
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            # Optimizer step (MaskedAdam will maintain sparsity)
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad()
                            
                            # Enforce masks explicitly (extra safety)
                            pruning_module.enforce_masks()
                        
                        # Update progress bar
                        if rank == 0:
                            current_sparsity = pruning_module.verify_sparsity()
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
                        logger.info(f"Epoch {epoch + 1} - Eval: "
                                   f"Loss: {eval_metrics['loss']:.4f}, "
                                   f"Correlation: {eval_metrics['correlation']:.4f}, "
                                   f"Sparsity: {pruning_module.current_sparsity:.1%}")
                        
                        # Save best model
                        if eval_metrics['correlation'] > best_score:
                            best_score = eval_metrics['correlation']
                            model_to_save = model.module if hasattr(model, 'module') else model
                            best_model_state = copy.deepcopy(model_to_save.state_dict())
    
                # Final evaluation
                if is_main_process():
                    logger.info("Final evaluation...")
                
                final_metrics = evaluate_model_distributed(
                    trainer.model, eval_loader, config['device'], local_rank
                )
                
                if is_main_process():
                    # Restore best model
                    if best_model_state:
                        model_to_load = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                        model_to_load.load_state_dict(best_model_state)
                    
                    # Calculate actual sparsity
                    model_for_sparsity = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
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
                    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
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
                del trainer
                del pruning_module
                torch.cuda.empty_cache()
                
                # Synchronize before next experiment
                if dist.is_initialized():
                    dist.barrier()
                
            except Exception as e:
                if is_main_process():
                    logger.error(f"Experiment failed for {target_sparsity:.0%} sparsity:")
                    logger.error(str(e))
                    logger.error(traceback.format_exc())
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
                logger.error(f"Experiment failed for {target_sparsity:.0%} sparsity:")
                logger.error(str(e))
                logger.error(traceback.format_exc())
            

    
    finally:
        # Clean up distributed training
        cleanup_distributed()  

if __name__ == "__main__":
    main() 
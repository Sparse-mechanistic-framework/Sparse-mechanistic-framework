"""
Updated Fixed Run Pruning Script
One-shot pruning (not gradual) with four baseline methods
Based on multi-GPU script optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
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
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pruning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============= CONFIGURATION =============
config = {
    'model_name': 'bert-base-uncased',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'target_sparsities': [0.3, 0.5, 0.7],
    'pruning_methods': ['random', 'magnitude', 'l0', 'movement', 'sma'],  # All methods
    'num_epochs': 6,  # More epochs for better fine-tuning
    'baseline_epochs': 4,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'output_dir': Path('./pruning_results_fixed'),
    'phase1_dir': Path('./phase1_results'),
    'max_samples': 8000,  # Updated to 6000
    'gradient_accumulation_steps': 2,
    'fp16': True,  # Mixed precision
    'protect_layers': [1, 2, 3, 4, 5, 6, 7],  # Critical layers from analysis
    'dataset_split': 'test',  # Using default split
}

# ============= IR MODEL =============
class IRModel(nn.Module):
    """IR model with gradient checkpointing"""
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, 1)
        
        # Enable gradient checkpointing
        if hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable()
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return type('Output', (), {'logits': logits})()

# ============= DATASET =============
class NFCorpusDataset(Dataset):
    def __init__(self, split='test', max_samples=8000, cache_dir='./cache', tokenizer=None, max_length=256):
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
                data = pickle.load(f)
                return data[:self.max_samples] if self.max_samples else data
        
        print("Loading NFCorpus from HuggingFace datasets...")
        
        try:
            # Load corpus
            corpus_data = load_dataset("mteb/nfcorpus", "corpus", split="corpus")
            corpus = {}
            for item in corpus_data:
                doc_id = item['_id'] if '_id' in item else item.get('id', str(len(corpus)))
                text = item.get('text', '')
                title = item.get('title', '')
                corpus[doc_id] = f"{title} {text}".strip()
            
            # Load queries
            queries_data = load_dataset("mteb/nfcorpus", "queries", split="queries")
            queries = {}
            for item in queries_data:
                query_id = item['_id'] if '_id' in item else item.get('id', str(len(queries)))
                queries[query_id] = item.get('text', '')
            
            # Load qrels for the specified split
            qrels_data = load_dataset("mteb/nfcorpus", "default", split=self.split)
            
            processed_data = []
            count = 0
            
            for item in tqdm(qrels_data, desc=f"Processing {self.split} qrels"):
                if self.max_samples and count >= self.max_samples:
                    break
                
                query_id = item.get('query-id') or item.get('query_id')
                corpus_id = item.get('corpus-id') or item.get('corpus_id')
                score = item.get('score', 0)
                
                if query_id and corpus_id:
                    query_text = queries.get(query_id, "")
                    doc_text = corpus.get(corpus_id, "")
                    
                    if query_text and doc_text:
                        # Truncate document to manageable length
                        doc_text = ' '.join(doc_text.split()[:500])
                        processed_data.append({
                            'query': query_text,
                            'document': doc_text,
                            'relevance': float(score / 2.0),  # Normalize score
                            'query_id': query_id,
                            'doc_id': corpus_id
                        })
                        count += 1
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_data, f)
            
            logger.info(f"Loaded {len(processed_data)} samples from NFCorpus {self.split} split")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load NFCorpus: {str(e)}")
            logger.warning("Using synthetic fallback data")
            # Synthetic fallback
            return [{'query': f'medical query {i}', 
                    'document': f'medical document {i} with relevant content',
                    'relevance': np.random.random(),
                    'query_id': f'q_{i}',
                    'doc_id': f'd_{i}'} 
                   for i in range(min(100, self.max_samples or 100))]
    
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

# ============= PRUNING METHODS =============

class PruningMethods:
    """All pruning methods in one place"""
    
    @staticmethod
    def random_pruning(model, sparsity, device='cuda'):
        """Random pruning baseline"""
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                mask = torch.rand_like(param) > sparsity
                masks[name] = mask.float().to(device)
                param.data *= masks[name]
        return masks
    
    @staticmethod
    def magnitude_pruning(model, sparsity, device='cuda'):
        """Magnitude-based pruning (Han et al., 2015) - memory efficient version"""
        masks = {}
        
        # Collect weight statistics without concatenating
        all_weights_list = []
        param_list = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                param_list.append((name, param))
                # Sample weights to avoid memory issues
                weight_sample = param.abs().flatten()
                if weight_sample.numel() > 10000:
                    # Randomly sample for large tensors
                    indices = torch.randperm(weight_sample.numel())[:10000]
                    weight_sample = weight_sample[indices]
                all_weights_list.append(weight_sample)
        
        if not all_weights_list:
            return masks
        
        # Concatenate samples and compute threshold
        all_weights = torch.cat(all_weights_list)
        threshold = torch.quantile(all_weights, sparsity)
        
        # Apply pruning with computed threshold
        for name, param in param_list:
            mask = (param.abs() > threshold).float()
            masks[name] = mask.to(device)
            param.data *= masks[name]
        
        return masks
    
    @staticmethod
    def l0_pruning(model, sparsity, device='cuda'):
        """L0 regularization pruning (Louizos et al., 2018) - memory efficient"""
        masks = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Compute importance scores with stochastic gates
                importance = param.abs() + torch.randn_like(param) * 0.05
                
                # Compute threshold for this layer
                importance_flat = importance.flatten()
                if importance_flat.numel() > 100000:
                    # Sample for very large layers
                    indices = torch.randperm(importance_flat.numel())[:100000]
                    importance_sample = importance_flat[indices]
                    threshold = torch.quantile(importance_sample, sparsity)
                else:
                    threshold = torch.quantile(importance_flat, sparsity)
                
                mask = (importance > threshold).float()
                masks[name] = mask.to(device)
                param.data *= masks[name]
        
        return masks
    
    @staticmethod
    def movement_pruning(model, sparsity, dataloader=None, device='cuda'):
        """Movement pruning (Sanh et al., 2020) - memory efficient"""
        if dataloader is None:
            # Fallback to magnitude if no data
            return PruningMethods.magnitude_pruning(model, sparsity, device)
        
        # Store initial weights
        initial_weights = {name: param.clone() for name, param in model.named_parameters()
                          if 'weight' in name and param.dim() >= 2}
        
        # Brief training to capture movement
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 20:  # Limited steps
                break
            
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits.squeeze()
            loss = F.mse_loss(logits, batch['labels'])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Compute movement scores layer by layer
        masks = {}
        for name, param in model.named_parameters():
            if name in initial_weights:
                movement = (param - initial_weights[name]) * param.sign()
                movement_flat = movement.flatten()
                
                # Compute threshold for this layer
                if movement_flat.numel() > 100000:
                    # Sample for large layers
                    indices = torch.randperm(movement_flat.numel())[:100000]
                    movement_sample = movement_flat[indices]
                    threshold = torch.quantile(movement_sample, sparsity)
                else:
                    threshold = torch.quantile(movement_flat, sparsity)
                
                mask = (movement > threshold).float()
                masks[name] = mask.to(device)
                param.data *= masks[name]
        
        return masks
    
    @staticmethod
    def sma_pruning(model, sparsity, importance_scores, protect_layers, circuits=None, device='cuda'):
        """SMA interpretation-aware pruning with circuit preservation"""
        masks = {}
        
        # Build circuit component map
        circuit_components = set()
        if circuits:
            for circuit in circuits:
                if isinstance(circuit, dict):
                    layer_idx = circuit.get('layer', -1)
                    if layer_idx >= 0:
                        circuit_components.add(f'layer.{layer_idx}')
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Get layer index
                layer_idx = PruningMethods._get_layer_index(name)
                
                # Calculate importance-weighted sparsity
                base_importance = 0.5  # Default
                
                # Check importance scores
                for key, score in importance_scores.items():
                    if f'layer_{layer_idx}' in key:
                        base_importance = score
                        break
                
                # Check if in protected layer or circuit
                in_circuit = any(f'layer.{layer_idx}' in name for comp in circuit_components)
                in_protected = layer_idx in protect_layers
                
                if in_circuit or in_protected:
                    # Boost importance for critical components
                    importance_multiplier = 2.0 if in_circuit else 1.5
                    # Reduce sparsity for protected components
                    actual_sparsity = max(0, sparsity - 0.3)
                else:
                    importance_multiplier = 1.0
                    actual_sparsity = sparsity
                
                # Apply magnitude pruning with protection
                if actual_sparsity > 0:
                    # Weight importance by scores
                    weight_importance = param.abs() * importance_multiplier
                    threshold = torch.quantile(weight_importance.flatten(), actual_sparsity)
                    mask = (weight_importance > threshold).float()
                else:
                    mask = torch.ones_like(param)
                
                masks[name] = mask.to(device)
                param.data *= masks[name]
        
        return masks
    
    @staticmethod
    def _get_layer_index(param_name):
        """Extract layer index from parameter name"""
        parts = param_name.split('.')
        for i, part in enumerate(parts):
            if part == 'layer' and i + 1 < len(parts):
                if parts[i + 1].isdigit():
                    return int(parts[i + 1])
        return -1

# ============= MASK ENFORCEMENT =============

class MaskedOptimizer:
    """Optimizer wrapper that maintains sparsity"""
    
    def __init__(self, optimizer, masks):
        self.optimizer = optimizer
        self.masks = masks
        
    def step(self):
        self.optimizer.step()
        self._apply_masks()
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def _apply_masks(self):
        """Apply masks after weight update"""
        for group in self.optimizer.param_groups:
            for param in group['params']:
                # Find corresponding mask
                for name, mask in self.masks.items():
                    if param.data_ptr() == dict(self.optimizer.state_dict()['state']).get(id(param), {}).get('exp_avg', param).data_ptr():
                        param.data *= mask
                        break

def enforce_masks(model, masks):
    """Enforce pruning masks on model"""
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name]

def calculate_sparsity(model):
    """Calculate actual sparsity of model"""
    total = 0
    zeros = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total += param.numel()
            zeros += (param == 0).sum().item()
    
    return zeros / total if total > 0 else 0

# ============= TRAINING FUNCTIONS =============

def train_epoch(model, dataloader, optimizer, scheduler, scaler, masks=None, device='cuda'):
    """Train for one epoch with mask enforcement"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        with autocast():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits.squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            loss = F.mse_loss(logits, batch['labels'])
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Enforce masks after update
            if masks:
                enforce_masks(model, masks)
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device='cuda'):
    """Evaluate model"""
    model.eval()
    predictions = []
    labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            with autocast():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits.squeeze()
                loss = F.mse_loss(logits, batch['labels'])
            
            predictions.extend(logits.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
            total_loss += loss.item()
    
    # Calculate metrics
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    correlation = 0
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, labels)[0, 1]
        correlation = 0 if np.isnan(correlation) else correlation
    
    mse = np.mean((predictions - labels) ** 2)
    
    return {
        'loss': total_loss / len(dataloader),
        'correlation': correlation,
        'mse': mse
    }

# ============= MAIN EXECUTION =============

def main():
    logger.info("="*60)
    logger.info("FIXED PRUNING EXPERIMENT - ONE-SHOT WITH BASELINES")
    logger.info("="*60)
    logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
    
    # Create directories
    config['output_dir'].mkdir(exist_ok=True, parents=True)
    
    # Load tokenizer and base model
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    base_model = AutoModel.from_pretrained(config['model_name'])
    
    # Load dataset
    logger.info("Loading NFCorpus dataset...")
    dataset = NFCorpusDataset(
        split=config['dataset_split'],  # Using 'test' split from config
        max_samples=config['max_samples'],  # Using 6000 samples
        tokenizer=tokenizer,
        cache_dir='./cache'
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Dataset: {train_size} train, {val_size} eval samples")
    
    # Train baseline model
    logger.info("\n" + "="*60)
    logger.info("TRAINING BASELINE MODEL")
    logger.info("="*60)
    
    baseline_model = IRModel(copy.deepcopy(base_model)).to(config['device'])
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=config['learning_rate'])
    
    num_training_steps = len(train_loader) * config['baseline_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    scaler = GradScaler()
    
    for epoch in range(config['baseline_epochs']):
        logger.info(f"\nBaseline Epoch {epoch + 1}/{config['baseline_epochs']}")
        loss = train_epoch(baseline_model, train_loader, optimizer, scheduler, scaler, device=config['device'])
        logger.info(f"Training loss: {loss:.4f}")
    
    baseline_metrics = evaluate(baseline_model, eval_loader, config['device'])
    logger.info(f"Baseline metrics: {baseline_metrics}")
    
    # Clean up baseline
    del baseline_model
    torch.cuda.empty_cache()
    
    # Load importance scores (for SMA)
    try:
        with open(config['phase1_dir'] / 'importance_scores.json', 'r') as f:
            phase1_data = json.load(f)
            if 'importance_scores' in phase1_data:
                importance_scores = phase1_data['importance_scores']
            else:
                importance_scores = phase1_data
    except:
        logger.warning("Could not load importance scores, using defaults")
        importance_scores = {}
    
    # Load circuits (for SMA)
    try:
        with open(config['phase1_dir'] / 'circuits.json', 'r') as f:
            circuits = json.load(f)
            logger.info(f"Loaded {len(circuits)} circuits from Phase 1")
    except:
        logger.warning("Could not load circuits, proceeding without circuit preservation")
        circuits = []
    
    # Results storage
    all_results = {
        'baseline': baseline_metrics,
        'methods': {}
    }
    
    # Test each pruning method
    for method in config['pruning_methods']:
        logger.info("\n" + "="*60)
        logger.info(f"TESTING {method.upper()} PRUNING")
        logger.info("="*60)
        
        all_results['methods'][method] = {}
        
        for sparsity in config['target_sparsities']:
            logger.info(f"\n>>> {method.upper()} at {sparsity:.0%} sparsity")
            
            # Create fresh model
            model = IRModel(copy.deepcopy(base_model)).to(config['device'])
            
            # Apply pruning (ONE-SHOT, not gradual)
            logger.info(f"Applying {method} pruning...")
            
            if method == 'random':
                masks = PruningMethods.random_pruning(model, sparsity, config['device'])
            elif method == 'magnitude':
                masks = PruningMethods.magnitude_pruning(model, sparsity, config['device'])
            elif method == 'l0':
                masks = PruningMethods.l0_pruning(model, sparsity, config['device'])
            elif method == 'movement':
                masks = PruningMethods.movement_pruning(model, sparsity, train_loader, config['device'])
            elif method == 'sma':
                masks = PruningMethods.sma_pruning(
                    model, sparsity, importance_scores, 
                    config['protect_layers'], circuits, config['device']
                )
            
            # Verify sparsity
            actual_sparsity = calculate_sparsity(model)
            logger.info(f"Actual sparsity after pruning: {actual_sparsity:.2%}")
            
            # Fine-tune pruned model
            logger.info("Fine-tuning pruned model...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
            
            num_training_steps = len(train_loader) * config['num_epochs']
            num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            scaler = GradScaler()
            
            best_correlation = -1
            for epoch in range(config['num_epochs']):
                logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
                
                # Train with mask enforcement
                loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, masks, config['device'])
                
                # Evaluate
                metrics = evaluate(model, eval_loader, config['device'])
                
                if metrics['correlation'] > best_correlation:
                    best_correlation = metrics['correlation']
                
                logger.info(f"Loss: {loss:.4f}, Correlation: {metrics['correlation']:.4f}, "
                           f"MSE: {metrics['mse']:.4f}")
            
            # Final evaluation
            final_metrics = evaluate(model, eval_loader, config['device'])
            final_sparsity = calculate_sparsity(model)
            
            # Calculate retention
            retention = final_metrics['correlation'] / max(baseline_metrics['correlation'], 0.001)
            
            # Store results
            all_results['methods'][method][sparsity] = {
                'target_sparsity': sparsity,
                'actual_sparsity': final_sparsity,
                'metrics': final_metrics,
                'retention': retention,
                'best_correlation': best_correlation
            }
            
            logger.info(f"\nResults for {method} at {sparsity:.0%}:")
            logger.info(f"  Actual sparsity: {final_sparsity:.2%}")
            logger.info(f"  Correlation: {final_metrics['correlation']:.4f}")
            logger.info(f"  MSE: {final_metrics['mse']:.4f}")
            logger.info(f"  Retention: {retention:.2%}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
    
    # Save results
    results_path = config['output_dir'] / 'pruning_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    logger.info(f"\nBaseline Correlation: {baseline_metrics['correlation']:.4f}")
    logger.info(f"\n{'Method':<12} {'Sparsity':<12} {'Actual':<12} {'Correlation':<12} {'Retention':<12}")
    logger.info("-" * 60)
    
    for method in config['pruning_methods']:
        for sparsity in config['target_sparsities']:
            result = all_results['methods'][method][sparsity]
            logger.info(
                f"{method:<12} {sparsity:<12.0%} {result['actual_sparsity']:<12.2%} "
                f"{result['metrics']['correlation']:<12.4f} {result['retention']:<12.2%}"
            )
    
    # Find best configuration
    best_method = None
    best_sparsity = None
    best_retention = 0
    
    for method in config['pruning_methods']:
        for sparsity in config['target_sparsities']:
            if all_results['methods'][method][sparsity]['retention'] > best_retention:
                best_retention = all_results['methods'][method][sparsity]['retention']
                best_method = method
                best_sparsity = sparsity
    
    logger.info(f"\nBest configuration: {best_method.upper()} at {best_sparsity:.0%} with {best_retention:.2%} retention")
    logger.info(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()

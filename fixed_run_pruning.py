"""
Updated Fixed Run Pruning Script
One-shot pruning (not gradual) with four baseline methods
Based on multi-GPU script optimizations
Updated with proper NFCorpus dataset handling
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

def is_main_process():
    """Check if this is the main process (for distributed training compatibility)"""
    return True  # Single process for now

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
    'max_samples': 6000,  # Updated to 6000
    'gradient_accumulation_steps': 2,
    'fp16': True,  # Mixed precision
    'protect_layers': [2, 3, 4, 5, 6],  # Critical layers from analysis
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
    def __init__(self, split='test', max_samples=6000, cache_dir='./cache', tokenizer=None, max_length=256):
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
                cached_data = pickle.load(f)
                if len(cached_data) >= self.max_samples:
                    return cached_data[:self.max_samples]
        
        # Only main process downloads data
        if is_main_process():
            print("Loading NFCorpus from HuggingFace datasets...")
            
            try:
                # Load corpus documents
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
                
                # Load relevance judgments (qrels)
                qrels_data = load_dataset("mteb/nfcorpus", "default", split=self.split)
                
                processed_data = []
                count = 0
                
                # Process qrels data to create query-document pairs
                for item in qrels_data:
                    if count >= self.max_samples:
                        break
                    
                    query_id = item.get('query-id', '')
                    corpus_id = item.get('corpus-id', '')
                    score = item.get('score', 0)
                    
                    # Get query and document text
                    query_text = queries.get(query_id, '')
                    doc_text = corpus.get(corpus_id, '')
                    
                    if query_text and doc_text:
                        processed_data.append({
                            'query': query_text,
                            'document': doc_text[:500],  # Truncate document
                            'relevance': float(score),
                            'query_id': query_id,
                            'doc_id': corpus_id
                        })
                        count += 1
                
                # If we still need more samples, create additional combinations
                if len(processed_data) < self.max_samples:
                    logger.info(f"Got {len(processed_data)} samples from qrels, generating additional samples to reach {self.max_samples}")
                    
                    # Get all available queries and documents
                    all_queries = list(queries.items())
                    all_docs = list(corpus.items())
                    
                    while len(processed_data) < self.max_samples:
                        # Randomly select query and document
                        query_id, query_text = all_queries[np.random.randint(0, len(all_queries))]
                        doc_id, doc_text = all_docs[np.random.randint(0, len(all_docs))]
                        
                        processed_data.append({
                            'query': query_text,
                            'document': doc_text[:500],
                            'relevance': np.random.random(),  # Random relevance for synthetic pairs
                            'query_id': query_id,
                            'doc_id': doc_id
                        })
                
                # Shuffle for variety
                np.random.shuffle(processed_data)
                
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                
                logger.info(f"Created {len(processed_data)} query-document pairs")
                return processed_data[:self.max_samples]
            
            except Exception as e:
                logger.warning(f"Failed to load NFCorpus: {e}. Using synthetic data.")
                # Synthetic fallback
                return [{'query': f'query {i}', 
                        'document': f'document {i}',
                        'relevance': np.random.random(),
                        'query_id': f'q{i}',
                        'doc_id': f'd{i}'} 
                       for i in range(self.max_samples)]
        
        # Wait for main process to finish (for distributed training)
        # For single process, just load the cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)[:self.max_samples]
        
        # Fallback
        return [{'query': f'query {i}', 
                'document': f'document {i}',
                'relevance': np.random.random(),
                'query_id': f'q{i}',
                'doc_id': f'd{i}'} 
               for i in range(self.max_samples)]
    
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
        """Magnitude-based pruning (Han et al., 2015)"""
        masks = {}
        
        # Collect all weights and compute threshold
        all_weights = []
        param_list = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                all_weights.append(param.abs().flatten())
                param_list.append((name, param))
        
        if not all_weights:
            return masks
            
        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights, sparsity)
        
        # Apply pruning
        for name, param in param_list:
            mask = (param.abs() > threshold).float()
            masks[name] = mask.to(device)
            param.data *= masks[name]
        
        return masks
    
    @staticmethod
    def l0_pruning(model, sparsity, device='cuda'):
        """L0 regularization pruning (Louizos et al., 2018)"""
        masks = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Compute importance scores with stochastic gates
                importance = param.abs() + torch.randn_like(param) * 0.05
                threshold = torch.quantile(importance.flatten(), sparsity)
                mask = (importance > threshold).float()
                masks[name] = mask.to(device)
                param.data *= masks[name]
        
        return masks
    
    @staticmethod
    def movement_pruning(model, sparsity, dataloader=None, device='cuda'):
        """Movement pruning (Sanh et al., 2020)"""
        if dataloader is None:
            # Fallback to magnitude if no data
            return PruningMethods.magnitude_pruning(model, sparsity, device)
        
        # Store initial weights
        initial_weights = {name: param.clone() for name, param in model.named_parameters()}
        
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
        
        # Compute movement scores
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                movement = (param - initial_weights[name]) * param.sign()
                threshold = torch.quantile(movement.flatten(), sparsity)
                mask = (movement > threshold).float()
                masks[name] = mask.to(device)
                param.data *= masks[name]
        
        return masks
    
    @staticmethod
    def sma_pruning(model, sparsity, importance_scores, protect_layers, device='cuda'):
        """SMA interpretation-aware pruning"""
        masks = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Get layer index
                layer_idx = PruningMethods._get_layer_index(name)
                
                # Check if in protected layer
                if layer_idx in protect_layers:
                    # Reduce sparsity for protected layers
                    actual_sparsity = max(0, sparsity - 0.3)
                else:
                    actual_sparsity = sparsity
                
                # Apply magnitude pruning with protection
                if actual_sparsity > 0:
                    threshold = torch.quantile(param.abs().flatten(), actual_sparsity)
                    mask = (param.abs() > threshold).float()
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
        split='test',
        max_samples=config['max_samples'],
        tokenizer=tokenizer
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
            importance_scores = json.load(f)
    except:
        logger.warning("Could not load importance scores, using defaults")
        importance_scores = {}
    
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
                    config['protect_layers'], config['device']
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

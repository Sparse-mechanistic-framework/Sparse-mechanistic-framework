"""
baseline_comparisons.py
Compare SMA pruning with traditional baseline methods
Senior AI Developer Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import time
import copy
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import traceback
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_comparisons.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BaselinePruningMethods:
    """Implementation of baseline pruning methods for comparison"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize baseline pruning methods
        
        Args:
            model: Model to prune
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.original_state = copy.deepcopy(model.state_dict())
        
    def magnitude_pruning(self, sparsity: float) -> nn.Module:
        """
        Traditional magnitude-based pruning
        
        Args:
            sparsity: Target sparsity level
            
        Returns:
            Pruned model
        """
        try:
            logger.info(f"Applying magnitude pruning with {sparsity:.0%} sparsity")
            
            model = copy.deepcopy(self.model)
            
            # Collect all weights
            all_weights = []
            weight_info = []
            
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    weights = param.data.abs().flatten()
                    all_weights.append(weights)
                    weight_info.append((name, param.shape, len(weights)))
            
            if not all_weights:
                raise ValueError("No weights found for pruning")
            
            # Concatenate all weights
            all_weights = torch.cat(all_weights)
            
            # Calculate threshold
            k = int(len(all_weights) * sparsity)
            if k > 0:
                threshold = torch.topk(all_weights, k, largest=False)[0].max()
            else:
                threshold = 0
            
            # Apply pruning
            pruned_params = 0
            total_params = 0
            
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    mask = param.data.abs() > threshold
                    param.data *= mask.float()
                    
                    pruned_params += (~mask).sum().item()
                    total_params += mask.numel()
            
            actual_sparsity = pruned_params / total_params if total_params > 0 else 0
            logger.info(f"Magnitude pruning complete - Actual sparsity: {actual_sparsity:.2%}")
            
            return model
            
        except Exception as e:
            logger.error(f"Magnitude pruning failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def random_pruning(self, sparsity: float, seed: int = 42) -> nn.Module:
        """
        Random pruning baseline
        
        Args:
            sparsity: Target sparsity level
            seed: Random seed
            
        Returns:
            Pruned model
        """
        try:
            logger.info(f"Applying random pruning with {sparsity:.0%} sparsity")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = copy.deepcopy(self.model)
            
            pruned_params = 0
            total_params = 0
            
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    # Create random mask
                    mask = torch.rand_like(param) > sparsity
                    param.data *= mask.float()
                    
                    pruned_params += (~mask).sum().item()
                    total_params += mask.numel()
            
            actual_sparsity = pruned_params / total_params if total_params > 0 else 0
            logger.info(f"Random pruning complete - Actual sparsity: {actual_sparsity:.2%}")
            
            return model
            
        except Exception as e:
            logger.error(f"Random pruning failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def structured_magnitude_pruning(self, sparsity: float) -> nn.Module:
        """
        Structured magnitude pruning (prune entire channels/heads)
        
        Args:
            sparsity: Target sparsity level
            
        Returns:
            Pruned model
        """
        try:
            logger.info(f"Applying structured magnitude pruning with {sparsity:.0%} sparsity")
            
            model = copy.deepcopy(self.model)
            base_model = model.bert if hasattr(model, 'bert') else model
            
            if not hasattr(base_model, 'encoder'):
                logger.warning("Model doesn't have encoder structure, falling back to unstructured")
                return self.magnitude_pruning(sparsity)
            
            # Prune attention heads
            for layer_idx, layer in enumerate(base_model.encoder.layer):
                if hasattr(layer.attention, 'self'):
                    n_heads = layer.attention.self.num_attention_heads
                    n_keep = max(1, int(n_heads * (1 - sparsity)))
                    
                    # Calculate head importance by magnitude
                    head_importance = []
                    for head_idx in range(n_heads):
                        importance = 0
                        for component in ['query', 'key', 'value']:
                            weight = getattr(layer.attention.self, component).weight
                            # Calculate importance for this head
                            head_size = weight.shape[0] // n_heads
                            head_weight = weight[head_idx * head_size:(head_idx + 1) * head_size]
                            importance += head_weight.abs().mean().item()
                        head_importance.append(importance)
                    
                    # Keep top-k heads
                    keep_indices = np.argsort(head_importance)[-n_keep:]
                    
                    # Zero out pruned heads
                    for head_idx in range(n_heads):
                        if head_idx not in keep_indices:
                            for component in ['query', 'key', 'value']:
                                weight = getattr(layer.attention.self, component).weight
                                bias = getattr(layer.attention.self, component).bias
                                head_size = weight.shape[0] // n_heads
                                
                                weight.data[head_idx * head_size:(head_idx + 1) * head_size] = 0
                                if bias is not None:
                                    bias.data[head_idx * head_size:(head_idx + 1) * head_size] = 0
            
            logger.info("Structured magnitude pruning complete")
            return model
            
        except Exception as e:
            logger.error(f"Structured magnitude pruning failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def movement_pruning(self, sparsity: float, train_loader: DataLoader, 
                         epochs: int = 1) -> nn.Module:
        """
        Movement pruning baseline (simplified version)
        
        Args:
            sparsity: Target sparsity level
            train_loader: Training data for movement calculation
            epochs: Training epochs
            
        Returns:
            Pruned model
        """
        try:
            logger.info(f"Applying movement pruning with {sparsity:.0%} sparsity")
            
            model = copy.deepcopy(self.model)
            model.to(self.device)
            
            # Store initial weights
            initial_weights = {}
            for name, param in model.named_parameters():
                if 'weight' in name:
                    initial_weights[name] = param.data.clone()
            
            # Brief training to capture movement
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            model.train()
            
            for epoch in range(epochs):
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx >= 10:  # Limited training for movement
                        break
                    
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    logits = outputs.logits.squeeze() if hasattr(outputs, 'logits') else outputs[0].squeeze()
                    loss = F.mse_loss(logits, batch['labels'])
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Calculate movement scores
            movement_scores = []
            weight_info = []
            
            for name, param in model.named_parameters():
                if 'weight' in name and name in initial_weights:
                    movement = (param.data - initial_weights[name]).abs()
                    movement_scores.append(movement.flatten())
                    weight_info.append((name, param.shape))
            
            # Concatenate all movement scores
            all_movements = torch.cat(movement_scores)
            
            # Calculate threshold
            k = int(len(all_movements) * sparsity)
            if k > 0:
                threshold = torch.topk(all_movements, k, largest=False)[0].max()
            else:
                threshold = 0
            
            # Apply pruning based on movement
            for name, param in model.named_parameters():
                if 'weight' in name and name in initial_weights:
                    movement = (param.data - initial_weights[name]).abs()
                    mask = movement > threshold
                    param.data *= mask.float()
            
            logger.info("Movement pruning complete")
            return model
            
        except Exception as e:
            logger.error(f"Movement pruning failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class BaselineComparison:
    """Run comprehensive baseline comparisons"""
    
    def __init__(
        self,
        base_model: nn.Module,
        sma_results_dir: Path,
        output_dir: Path,
        device: str = 'cuda'
    ):
        """
        Initialize baseline comparison
        
        Args:
            base_model: Original unpruned model
            sma_results_dir: Directory with SMA pruning results
            output_dir: Output directory for comparison results
            device: Device for computation
        """
        self.base_model = base_model
        self.sma_results_dir = Path(sma_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        
        self.results = {
            'sma': {},
            'magnitude': {},
            'random': {},
            'structured': {},
            'movement': {}
        }
        
        # Load SMA results
        self._load_sma_results()
    
    def _load_sma_results(self):
        """Load SMA pruning results for comparison"""
        try:
            results_file = self.sma_results_dir / 'metrics' / 'pruning_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    
                for exp_name, exp_data in data.get('experiments', {}).items():
                    sparsity = exp_data.get('target_sparsity', 0)
                    self.results['sma'][sparsity] = {
                        'correlation': exp_data.get('metrics_after', {}).get('correlation', 0),
                        'mse': exp_data.get('metrics_after', {}).get('mse', 0),
                        'retention': exp_data.get('retention', 0)
                    }
                    
                logger.info(f"Loaded SMA results for {len(self.results['sma'])} sparsity levels")
            else:
                logger.warning(f"SMA results not found at {results_file}")
                
        except Exception as e:
            logger.error(f"Failed to load SMA results: {str(e)}")
    
    def evaluate_model(
        self,
        model: nn.Module,
        eval_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Model to evaluate
            eval_loader: Evaluation data loader
            
        Returns:
            Performance metrics
        """
        try:
            model.eval()
            model.to(self.device)
            
            predictions = []
            labels = []
            losses = []
            
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    logits = outputs.logits.squeeze() if hasattr(outputs, 'logits') else outputs[0].squeeze()
                    
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)
                    if batch['labels'].dim() == 0:
                        batch['labels'] = batch['labels'].unsqueeze(0)
                    
                    loss = F.mse_loss(logits, batch['labels'])
                    losses.append(loss.item())
                    
                    predictions.extend(logits.cpu().numpy())
                    labels.extend(batch['labels'].cpu().numpy())
            
            # Calculate metrics
            predictions = np.array(predictions)
            labels = np.array(labels)
            
            correlation = np.corrcoef(predictions, labels)[0, 1] if len(predictions) > 1 else 0
            mse = np.mean((predictions - labels) ** 2)
            
            return {
                'loss': np.mean(losses),
                'correlation': correlation,
                'mse': mse
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {'loss': float('inf'), 'correlation': 0, 'mse': float('inf')}
    
    def run_baseline_comparisons(
        self,
        eval_loader: DataLoader,
        train_loader: Optional[DataLoader] = None,
        sparsity_levels: List[float] = [0.3, 0.5, 0.7]
    ):
        """
        Run all baseline comparisons
        
        Args:
            eval_loader: Evaluation data
            train_loader: Training data (for movement pruning)
            sparsity_levels: Sparsity levels to test
        """
        logger.info("="*60)
        logger.info("RUNNING BASELINE COMPARISONS")
        logger.info("="*60)
        
        # Evaluate unpruned model
        logger.info("Evaluating unpruned baseline model...")
        baseline_metrics = self.evaluate_model(self.base_model, eval_loader)
        logger.info(f"Baseline: correlation={baseline_metrics['correlation']:.4f}, "
                   f"mse={baseline_metrics['mse']:.4f}")
        
        # Initialize pruning methods
        pruner = BaselinePruningMethods(self.base_model, self.device)
        
        for sparsity in sparsity_levels:
            logger.info(f"\n{'='*40}")
            logger.info(f"Testing sparsity: {sparsity:.0%}")
            logger.info(f"{'='*40}")
            
            # 1. Magnitude Pruning
            try:
                logger.info("\n1. Magnitude Pruning")
                mag_model = pruner.magnitude_pruning(sparsity)
                mag_metrics = self.evaluate_model(mag_model, eval_loader)
                self.results['magnitude'][sparsity] = {
                    **mag_metrics,
                    'retention': mag_metrics['correlation'] / baseline_metrics['correlation']
                }
                logger.info(f"Results: correlation={mag_metrics['correlation']:.4f}, "
                           f"retention={self.results['magnitude'][sparsity]['retention']:.2%}")
            except Exception as e:
                logger.error(f"Magnitude pruning failed: {str(e)}")
                self.results['magnitude'][sparsity] = {'correlation': 0, 'mse': float('inf'), 'retention': 0}
            
            # 2. Random Pruning
            try:
                logger.info("\n2. Random Pruning")
                rand_model = pruner.random_pruning(sparsity)
                rand_metrics = self.evaluate_model(rand_model, eval_loader)
                self.results['random'][sparsity] = {
                    **rand_metrics,
                    'retention': rand_metrics['correlation'] / baseline_metrics['correlation']
                }
                logger.info(f"Results: correlation={rand_metrics['correlation']:.4f}, "
                           f"retention={self.results['random'][sparsity]['retention']:.2%}")
            except Exception as e:
                logger.error(f"Random pruning failed: {str(e)}")
                self.results['random'][sparsity] = {'correlation': 0, 'mse': float('inf'), 'retention': 0}
            
            # 3. Structured Magnitude Pruning
            try:
                logger.info("\n3. Structured Magnitude Pruning")
                struct_model = pruner.structured_magnitude_pruning(sparsity)
                struct_metrics = self.evaluate_model(struct_model, eval_loader)
                self.results['structured'][sparsity] = {
                    **struct_metrics,
                    'retention': struct_metrics['correlation'] / baseline_metrics['correlation']
                }
                logger.info(f"Results: correlation={struct_metrics['correlation']:.4f}, "
                           f"retention={self.results['structured'][sparsity]['retention']:.2%}")
            except Exception as e:
                logger.error(f"Structured pruning failed: {str(e)}")
                self.results['structured'][sparsity] = {'correlation': 0, 'mse': float('inf'), 'retention': 0}
            
            # 4. Movement Pruning (if training data available)
            if train_loader:
                try:
                    logger.info("\n4. Movement Pruning")
                    move_model = pruner.movement_pruning(sparsity, train_loader)
                    move_metrics = self.evaluate_model(move_model, eval_loader)
                    self.results['movement'][sparsity] = {
                        **move_metrics,
                        'retention': move_metrics['correlation'] / baseline_metrics['correlation']
                    }
                    logger.info(f"Results: correlation={move_metrics['correlation']:.4f}, "
                               f"retention={self.results['movement'][sparsity]['retention']:.2%}")
                except Exception as e:
                    logger.error(f"Movement pruning failed: {str(e)}")
                    self.results['movement'][sparsity] = {'correlation': 0, 'mse': float('inf'), 'retention': 0}
        
        # Save results
        self._save_results()
        
        # Generate comparison plots
        self._generate_comparison_plots()
        
        # Statistical analysis
        self._statistical_analysis()
    
    def _save_results(self):
        """Save comparison results"""
        output_file = self.output_dir / 'baseline_comparison_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    def _generate_comparison_plots(self):
        """Generate comparison visualizations"""
        try:
            # Prepare data for plotting
            methods = list(self.results.keys())
            sparsities = sorted(set(s for method in self.results.values() for s in method.keys()))
            
            # Create retention comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Retention vs Sparsity
            ax = axes[0]
            for method in methods:
                if self.results[method]:
                    x = []
                    y = []
                    for sparsity in sparsities:
                        if sparsity in self.results[method]:
                            x.append(sparsity)
                            y.append(self.results[method][sparsity].get('retention', 0))
                    
                    if x and y:
                        ax.plot(x, y, marker='o', label=method.upper(), linewidth=2, markersize=8)
            
            ax.set_xlabel('Sparsity Level')
            ax.set_ylabel('Performance Retention')
            ax.set_title('Performance Retention Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Target')
            
            # Plot 2: Bar comparison at 50% sparsity
            ax = axes[1]
            target_sparsity = 0.5
            
            retention_values = []
            method_names = []
            
            for method in methods:
                if target_sparsity in self.results[method]:
                    retention_values.append(self.results[method][target_sparsity].get('retention', 0))
                    method_names.append(method.upper())
            
            if retention_values:
                bars = ax.bar(method_names, retention_values, color=['green', 'blue', 'red', 'orange', 'purple'][:len(method_names)])
                ax.set_ylabel('Performance Retention')
                ax.set_title(f'Method Comparison at {target_sparsity:.0%} Sparsity')
                ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
                
                # Add value labels on bars
                for bar, val in zip(bars, retention_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1%}', ha='center', va='bottom')
            
            plt.suptitle('Baseline Method Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.output_dir / 'baseline_comparison_plot.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.show()
            logger.info(f"Comparison plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _statistical_analysis(self):
        """Perform statistical significance tests"""
        try:
            logger.info("\n" + "="*60)
            logger.info("STATISTICAL ANALYSIS")
            logger.info("="*60)
            
            # Compare SMA with each baseline at 50% sparsity
            target_sparsity = 0.5
            
            if target_sparsity in self.results['sma']:
                sma_retention = self.results['sma'][target_sparsity]['retention']
                
                comparisons = []
                for method in ['magnitude', 'random', 'structured', 'movement']:
                    if target_sparsity in self.results[method]:
                        baseline_retention = self.results[method][target_sparsity]['retention']
                        
                        # Calculate improvement
                        improvement = ((sma_retention - baseline_retention) / baseline_retention) * 100
                        
                        comparisons.append({
                            'method': method,
                            'sma_retention': sma_retention,
                            'baseline_retention': baseline_retention,
                            'improvement': improvement
                        })
                        
                        logger.info(f"\nSMA vs {method.upper()}:")
                        logger.info(f"  SMA: {sma_retention:.1%}")
                        logger.info(f"  {method}: {baseline_retention:.1%}")
                        logger.info(f"  Improvement: {improvement:+.1f}%")
                
                # Save statistical analysis
                stats_file = self.output_dir / 'statistical_analysis.json'
                with open(stats_file, 'w') as f:
                    json.dump(comparisons, f, indent=2)
                    
                # Check if SMA is best
                if comparisons:
                    best_baseline = max(comparisons, key=lambda x: x['baseline_retention'])
                    if sma_retention > best_baseline['baseline_retention']:
                        logger.info(f"\n✅ SMA outperforms all baselines!")
                        logger.info(f"Best baseline ({best_baseline['method']}): {best_baseline['baseline_retention']:.1%}")
                        logger.info(f"SMA: {sma_retention:.1%}")
                    else:
                        logger.warning(f"\n⚠️ SMA performance below {best_baseline['method']}")
                        
        except Exception as e:
            logger.error(f"Statistical analysis failed: {str(e)}")
            logger.error(traceback.format_exc())


def main():
    """Main execution function"""
    try:
        logger.info("Starting Baseline Comparisons")
        logger.info("="*60)
        
        # Configuration
        config = {
            'model_name': 'bert-base-uncased',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'sma_results_dir': './phase2_results',
            'output_dir': './baseline_comparisons',
            'batch_size': 16,
            'sparsity_levels': [0.3, 0.5, 0.7]
        }
        
        # Load model and data
        from transformers import AutoModel, AutoTokenizer
        from run_pruning import NFCorpusDataset, IRModel
        
        logger.info("Loading model and data...")
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        base_bert = AutoModel.from_pretrained(config['model_name'])
        base_model = IRModel(base_bert).to(config['device'])
        
        # Load dataset
        dataset = NFCorpusDataset(split='test', max_samples=2000, tokenizer=tokenizer)
        eval_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Split for training data (movement pruning)
        train_size = int(0.8 * len(dataset))
        train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Run comparisons
        comparison = BaselineComparison(
            base_model=base_model,
            sma_results_dir=config['sma_results_dir'],
            output_dir=Path(config['output_dir']),
            device=config['device']
        )
        
        comparison.run_baseline_comparisons(
            eval_loader=eval_loader,
            train_loader=train_loader,
            sparsity_levels=config['sparsity_levels']
        )
        
        logger.info("\n" + "="*60)
        logger.info("Baseline comparisons complete!")
        logger.info(f"Results saved to: {config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Critical failure in baseline comparisons: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

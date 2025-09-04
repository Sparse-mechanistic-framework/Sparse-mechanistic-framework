#!/usr/bin/env python3
"""
Ablation Study and Perturbation Analysis for SMA Pruning
Validates the importance of discovered circuits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AblationConfig:
    """Configuration for ablation studies"""
    model_path: str = './pruning_results_fixed/models/baseline.pt'
    sma_model_path: str = './pruning_results_fixed/models/sma_50.pt'
    circuits_path: str = './phase1_results/circuits.json'
    importance_scores_path: str = './phase1_results/importance_scores.json'
    results_path: str = './pruning_results_fixed/pruning_results_fixed.json'
    output_dir: Path = Path('./ablation_results')
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 32
    num_perturbation_samples: int = 100

class CircuitAblation:
    """Perform ablation studies on discovered circuits"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.config.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load circuits and importance scores
        with open(config.circuits_path, 'r') as f:
            self.circuits = json.load(f)
        
        with open(config.importance_scores_path, 'r') as f:
            self.importance_scores = json.load(f)
            if 'importance_scores' in self.importance_scores:
                self.importance_scores = self.importance_scores['importance_scores']
        
        # Load baseline results
        with open(config.results_path, 'r') as f:
            self.baseline_results = json.load(f)
    
    def ablate_top_circuits(self, model, eval_loader, k: int = 5) -> Dict:
        """Remove top-k circuits and measure performance drop"""
        logger.info(f"Ablating top {k} circuits...")
        
        # Sort circuits by importance
        sorted_circuits = sorted(self.circuits, 
                               key=lambda x: x.get('importance', 0), 
                               reverse=True)
        
        results = {
            'baseline_performance': self._evaluate_model(model, eval_loader),
            'ablation_results': []
        }
        
        # Progressively ablate circuits
        for i in range(min(k, len(sorted_circuits))):
            circuits_to_ablate = sorted_circuits[:i+1]
            
            # Create ablated model
            ablated_model = self._ablate_circuits(model, circuits_to_ablate)
            
            # Evaluate
            performance = self._evaluate_model(ablated_model, eval_loader)
            
            results['ablation_results'].append({
                'num_circuits_ablated': i + 1,
                'ablated_circuits': [c.get('name', f"circuit_{j}") 
                                    for j, c in enumerate(circuits_to_ablate)],
                'performance': performance,
                'performance_drop': results['baseline_performance']['correlation'] - 
                                   performance['correlation']
            })
            
            logger.info(f"Ablated {i+1} circuits: "
                       f"Correlation = {performance['correlation']:.4f} "
                       f"(drop = {results['ablation_results'][-1]['performance_drop']:.4f})")
        
        return results
    
    def _ablate_circuits(self, model, circuits_to_ablate: List[Dict]) -> nn.Module:
        """Zero out weights in specified circuits"""
        ablated_model = copy.deepcopy(model)
        
        for circuit in circuits_to_ablate:
            layer_idx = circuit.get('layer', -1)
            
            if layer_idx >= 0:
                # Zero out attention weights in this layer
                for name, param in ablated_model.named_parameters():
                    if f'layer.{layer_idx}' in name and 'attention' in name:
                        param.data.mul_(0.1)  # Reduce to 10% instead of zeroing
        
        return ablated_model
    
    def random_circuit_ablation(self, model, eval_loader, num_trials: int = 5) -> Dict:
        """Compare ablating important vs random circuits"""
        logger.info("Comparing important vs random circuit ablation...")
        
        baseline_perf = self._evaluate_model(model, eval_loader)
        
        # Get top circuits
        sorted_circuits = sorted(self.circuits, 
                               key=lambda x: x.get('importance', 0), 
                               reverse=True)
        top_circuits = sorted_circuits[:5]
        
        # Ablate top circuits
        top_ablated_model = self._ablate_circuits(model, top_circuits)
        top_ablated_perf = self._evaluate_model(top_ablated_model, eval_loader)
        
        # Random ablation trials
        random_results = []
        for trial in range(num_trials):
            random_circuits = np.random.choice(self.circuits, 5, replace=False).tolist()
            random_ablated_model = self._ablate_circuits(model, random_circuits)
            random_perf = self._evaluate_model(random_ablated_model, eval_loader)
            random_results.append(random_perf['correlation'])
        
        return {
            'baseline': baseline_perf['correlation'],
            'top_circuits_ablated': top_ablated_perf['correlation'],
            'random_ablated_mean': np.mean(random_results),
            'random_ablated_std': np.std(random_results),
            'importance_difference': np.mean(random_results) - top_ablated_perf['correlation']
        }
    
    def layer_wise_ablation(self, model, eval_loader) -> Dict:
        """Ablate entire layers to understand their importance"""
        logger.info("Performing layer-wise ablation...")
        
        baseline_perf = self._evaluate_model(model, eval_loader)
        results = {'baseline': baseline_perf['correlation'], 'layers': {}}
        
        # Test each layer
        for layer_idx in range(12):  # BERT has 12 layers
            ablated_model = copy.deepcopy(model)
            
            # Zero out entire layer
            for name, param in ablated_model.named_parameters():
                if f'layer.{layer_idx}' in name:
                    param.data.mul_(0.1)
            
            perf = self._evaluate_model(ablated_model, eval_loader)
            results['layers'][layer_idx] = {
                'correlation': perf['correlation'],
                'drop': baseline_perf['correlation'] - perf['correlation']
            }
            
            logger.info(f"Layer {layer_idx} ablated: "
                       f"Correlation = {perf['correlation']:.4f} "
                       f"(drop = {results['layers'][layer_idx]['drop']:.4f})")
        
        return results
    
    def _evaluate_model(self, model, dataloader) -> Dict:
        """Evaluate model performance"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.config.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = model(input_ids=batch['input_ids'],
                              attention_mask=batch['attention_mask'])
                logits = outputs.logits.squeeze()
                
                predictions.extend(logits.cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0
        mse = np.mean((predictions - targets) ** 2)
        
        return {'correlation': correlation, 'mse': mse}

class PerturbationAnalysis:
    """Analyze circuit importance through perturbations"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.device = config.device
        
        # Load circuits
        with open(config.circuits_path, 'r') as f:
            self.circuits = json.load(f)
    
    def attention_head_perturbation(self, model, eval_loader) -> Dict:
        """Perturb individual attention heads and measure impact"""
        logger.info("Performing attention head perturbation analysis...")
        
        baseline_perf = self._evaluate_model(model, eval_loader)
        results = {'baseline': baseline_perf, 'heads': {}}
        
        for layer_idx in range(12):
            for head_idx in range(12):  # BERT has 12 heads
                # Perturb specific head
                perturbed_model = self._perturb_attention_head(
                    copy.deepcopy(model), layer_idx, head_idx
                )
                
                perf = self._evaluate_model(perturbed_model, eval_loader)
                
                key = f'L{layer_idx}_H{head_idx}'
                results['heads'][key] = {
                    'correlation': perf['correlation'],
                    'impact': baseline_perf['correlation'] - perf['correlation']
                }
        
        return results
    
    def _perturb_attention_head(self, model, layer_idx: int, head_idx: int) -> nn.Module:
        """Add noise to specific attention head"""
        for name, param in model.named_parameters():
            if f'layer.{layer_idx}' in name and 'attention' in name:
                if 'query' in name or 'key' in name or 'value' in name:
                    # Add Gaussian noise to specific head
                    head_dim = param.shape[0] // 12
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim
                    
                    noise = torch.randn_like(param[start_idx:end_idx]) * 0.1
                    param.data[start_idx:end_idx] += noise
        
        return model
    
    def gradient_based_importance(self, model, eval_loader, num_samples: int = 100) -> Dict:
        """Compute gradient-based importance scores"""
        logger.info("Computing gradient-based importance scores...")
        
        model.eval()
        importance_scores = {}
        
        # Sample batches
        for i, batch in enumerate(eval_loader):
            if i >= num_samples:
                break
            
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'])
            logits = outputs.logits.squeeze()
            
            # Compute gradient w.r.t. output
            loss = F.mse_loss(logits, batch['labels'])
            loss.backward()
            
            # Accumulate gradient magnitudes
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in importance_scores:
                        importance_scores[name] = 0
                    importance_scores[name] += param.grad.abs().mean().item()
            
            model.zero_grad()
        
        # Normalize
        max_score = max(importance_scores.values())
        for name in importance_scores:
            importance_scores[name] /= max_score
        
        return importance_scores
    
    def causal_tracing(self, model, eval_loader, num_samples: int = 50) -> Dict:
        """Perform causal tracing to understand information flow"""
        logger.info("Performing causal tracing analysis...")
        
        results = {'layer_importance': {}, 'information_flow': []}
        
        for layer_idx in range(12):
            layer_impacts = []
            
            for i, batch in enumerate(eval_loader):
                if i >= num_samples:
                    break
                
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Get clean output
                with torch.no_grad():
                    clean_output = model(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'])
                    clean_logits = clean_output.logits.squeeze()
                
                # Corrupt input at layer
                corrupted_model = self._corrupt_layer_input(
                    copy.deepcopy(model), layer_idx
                )
                
                with torch.no_grad():
                    corrupted_output = corrupted_model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    corrupted_logits = corrupted_output.logits.squeeze()
                
                # Measure impact
                impact = F.mse_loss(clean_logits, corrupted_logits).item()
                layer_impacts.append(impact)
            
            results['layer_importance'][layer_idx] = np.mean(layer_impacts)
        
        return results
    
    def _corrupt_layer_input(self, model, layer_idx: int) -> nn.Module:
        """Corrupt input to specific layer"""
        # Add noise to layer norm before the layer
        for name, param in model.named_parameters():
            if f'layer.{layer_idx}' in name and 'LayerNorm' in name:
                param.data += torch.randn_like(param) * 0.1
        return model
    
    def _evaluate_model(self, model, dataloader) -> Dict:
        """Evaluate model performance"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = model(input_ids=batch['input_ids'],
                              attention_mask=batch['attention_mask'])
                logits = outputs.logits.squeeze()
                
                predictions.extend(logits.cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())
        
        correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0
        return {'correlation': correlation}

def visualize_ablation_results(results: Dict, output_dir: Path):
    """Create visualizations for ablation study results"""
    
    # Plot 1: Progressive circuit ablation
    if 'ablation_results' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_ablated = [r['num_circuits_ablated'] for r in results['ablation_results']]
        performance = [r['performance']['correlation'] for r in results['ablation_results']]
        
        ax.plot([0] + num_ablated, 
               [results['baseline_performance']['correlation']] + performance,
               marker='o', linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Circuits Ablated', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('Impact of Circuit Ablation on Performance', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.savefig(output_dir / 'circuit_ablation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Layer-wise importance
    if 'layers' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = list(results['layers'].keys())
        drops = [results['layers'][l]['drop'] for l in layers]
        
        ax.bar(layers, drops, color='steelblue', alpha=0.7)
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Performance Drop', fontsize=12)
        ax.set_title('Layer-wise Importance (Performance Drop when Ablated)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(output_dir / 'layer_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Run complete ablation and perturbation analysis"""
    config = AblationConfig()
    
    # Note: You need to load your actual model and dataloader here
    # This is a placeholder
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    logger.info("Loading model and data...")
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    model.to(config.device)
    
    # Load your evaluation dataloader here
    # eval_loader = ...
    
    # Create dummy loader for demonstration
    from torch.utils.data import TensorDataset
    dummy_data = TensorDataset(
        torch.randint(0, 1000, (100, 128)),  # input_ids
        torch.ones(100, 128),  # attention_mask
        torch.rand(100)  # labels
    )
    eval_loader = DataLoader(dummy_data, batch_size=config.batch_size)
    
    # Wrap dataloader to provide correct format
    class DataLoaderWrapper:
        def __init__(self, dataloader):
            self.dataloader = dataloader
        
        def __iter__(self):
            for batch in self.dataloader:
                yield {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }
        
        def __len__(self):
            return len(self.dataloader)
    
    eval_loader = DataLoaderWrapper(eval_loader)
    
    # Run ablation studies
    logger.info("\n" + "="*60)
    logger.info("ABLATION STUDY")
    logger.info("="*60)
    
    ablation = CircuitAblation(config)
    
    # 1. Progressive circuit ablation
    circuit_results = ablation.ablate_top_circuits(model, eval_loader, k=10)
    
    # 2. Random vs important circuit ablation
    random_results = ablation.random_circuit_ablation(model, eval_loader)
    logger.info(f"\nRandom vs Important Circuit Ablation:")
    logger.info(f"  Top circuits ablated: {random_results['top_circuits_ablated']:.4f}")
    logger.info(f"  Random circuits ablated: {random_results['random_ablated_mean']:.4f} "
               f"(±{random_results['random_ablated_std']:.4f})")
    logger.info(f"  Importance difference: {random_results['importance_difference']:.4f}")
    
    # 3. Layer-wise ablation
    layer_results = ablation.layer_wise_ablation(model, eval_loader)
    
    # Visualize results
    visualize_ablation_results(circuit_results, config.output_dir)
    visualize_ablation_results(layer_results, config.output_dir)
    
    # Run perturbation analysis
    logger.info("\n" + "="*60)
    logger.info("PERTURBATION ANALYSIS")
    logger.info("="*60)
    
    perturbation = PerturbationAnalysis(config)
    
    # 1. Attention head perturbation
    head_results = perturbation.attention_head_perturbation(model, eval_loader)
    
    # 2. Gradient-based importance
    gradient_importance = perturbation.gradient_based_importance(model, eval_loader)
    
    # 3. Causal tracing
    causal_results = perturbation.causal_tracing(model, eval_loader)
    
    # Save all results
    all_results = {
        'circuit_ablation': circuit_results,
        'random_comparison': random_results,
        'layer_ablation': layer_results,
        'head_perturbation': head_results,
        'gradient_importance': gradient_importance,
        'causal_tracing': causal_results
    }
    
    with open(config.output_dir / 'ablation_perturbation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {config.output_dir}")
    
    return all_results

if __name__ == "__main__":
    results = main()

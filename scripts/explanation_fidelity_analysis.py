"""
explanation_fidelity_analysis.py
Analyze how well explanations (circuits, attention patterns) are preserved after pruning
Senior AI Developer Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import traceback
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('explanation_fidelity.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ExplanationFidelityAnalyzer:
    """Analyze preservation of explanations in pruned models"""
    
    def __init__(
        self,
        phase1_dir: Path,
        phase2_dir: Path,
        output_dir: Path,
        device: str = 'cuda'
    ):
        """
        Initialize fidelity analyzer
        
        Args:
            phase1_dir: Directory with Phase 1 results
            phase2_dir: Directory with Phase 2 results
            output_dir: Output directory for analysis
            device: Device for computation
        """
        self.phase1_dir = Path(phase1_dir)
        self.phase2_dir = Path(phase2_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        
        # Load Phase 1 results
        self.original_circuits = self._load_circuits()
        self.original_importance = self._load_importance_scores()
        
        # Results storage
        self.fidelity_results = {
            'circuit_preservation': {},
            'attention_similarity': {},
            'importance_correlation': {},
            'activation_similarity': {}
        }
    
    def _load_circuits(self) -> List[Dict]:
        """Load original circuits from Phase 1"""
        try:
            circuits_file = self.phase1_dir / 'circuits.json'
            if circuits_file.exists():
                with open(circuits_file, 'r') as f:
                    circuits = json.load(f)
                logger.info(f"Loaded {len(circuits)} circuits from Phase 1")
                return circuits
            else:
                logger.warning(f"Circuits file not found at {circuits_file}")
                return []
        except Exception as e:
            logger.error(f"Failed to load circuits: {str(e)}")
            return []
    
    def _load_importance_scores(self) -> Dict[str, float]:
        """Load original importance scores from Phase 1"""
        try:
            importance_file = self.phase1_dir / 'importance_scores.json'
            if importance_file.exists():
                with open(importance_file, 'r') as f:
                    data = json.load(f)
                    scores = data.get('importance_scores', {})
                logger.info(f"Loaded {len(scores)} importance scores from Phase 1")
                return scores
            else:
                logger.warning(f"Importance scores file not found at {importance_file}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load importance scores: {str(e)}")
            return {}
    
    def analyze_circuit_preservation(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module,
        test_data: List[Tuple[str, str]],
        model_name: str
    ) -> Dict[str, Any]:
        """
        Analyze how well circuits are preserved in pruned model
        
        Args:
            original_model: Original unpruned model
            pruned_model: Pruned model
            test_data: Test query-document pairs
            model_name: Name of pruned model
            
        Returns:
            Circuit preservation metrics
        """
        try:
            logger.info(f"Analyzing circuit preservation for {model_name}")
            
            preservation_scores = []
            circuit_activations_original = []
            circuit_activations_pruned = []
            
            # Analyze each circuit
            for circuit in self.original_circuits[:20]:  # Analyze top 20 circuits
                circuit_name = circuit.get('name', 'unknown')
                layers = circuit.get('layers', [])
                components = circuit.get('components', [])
                
                # Check if circuit components still exist and are active
                is_preserved = True
                activation_ratio = []
                
                for layer_idx in layers:
                    for component in components:
                        # Get weights from both models
                        orig_weight = self._get_component_weight(original_model, layer_idx, component)
                        pruned_weight = self._get_component_weight(pruned_model, layer_idx, component)
                        
                        if orig_weight is not None and pruned_weight is not None:
                            # Check if component is pruned (all zeros)
                            if torch.allclose(pruned_weight, torch.zeros_like(pruned_weight)):
                                is_preserved = False
                            
                            # Calculate activation preservation ratio
                            orig_norm = torch.norm(orig_weight).item()
                            pruned_norm = torch.norm(pruned_weight).item()
                            
                            if orig_norm > 0:
                                ratio = pruned_norm / orig_norm
                                activation_ratio.append(ratio)
                
                # Calculate preservation score for this circuit
                if activation_ratio:
                    avg_ratio = np.mean(activation_ratio)
                    preservation_scores.append({
                        'circuit': circuit_name,
                        'preserved': is_preserved,
                        'activation_ratio': avg_ratio,
                        'importance': circuit.get('importance_score', 0)
                    })
            
            # Calculate overall metrics
            preserved_count = sum(1 for s in preservation_scores if s['preserved'])
            total_circuits = len(preservation_scores)
            preservation_rate = preserved_count / total_circuits if total_circuits > 0 else 0
            
            # Weighted preservation by importance
            weighted_preservation = 0
            total_importance = 0
            for score in preservation_scores:
                weight = score['importance']
                weighted_preservation += score['activation_ratio'] * weight
                total_importance += weight
            
            if total_importance > 0:
                weighted_preservation /= total_importance
            
            # High-importance circuit preservation
            high_importance_circuits = [s for s in preservation_scores if s['importance'] > 50]
            high_importance_preserved = sum(1 for s in high_importance_circuits if s['preserved'])
            high_importance_rate = (high_importance_preserved / len(high_importance_circuits) 
                                   if high_importance_circuits else 0)
            
            results = {
                'model': model_name,
                'total_circuits_analyzed': total_circuits,
                'circuits_preserved': preserved_count,
                'preservation_rate': preservation_rate,
                'weighted_preservation': weighted_preservation,
                'high_importance_preservation': high_importance_rate,
                'circuit_details': preservation_scores
            }
            
            logger.info(f"Circuit preservation rate: {preservation_rate:.2%}")
            logger.info(f"Weighted preservation: {weighted_preservation:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Circuit preservation analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _get_component_weight(
        self,
        model: nn.Module,
        layer_idx: int,
        component: str
    ) -> Optional[torch.Tensor]:
        """Get weight tensor for specific component"""
        try:
            base_model = model.bert if hasattr(model, 'bert') else model
            
            if hasattr(base_model, 'encoder') and layer_idx < len(base_model.encoder.layer):
                layer = base_model.encoder.layer[layer_idx]
                
                if 'attention' in component:
                    if hasattr(layer.attention, 'self'):
                        return layer.attention.self.query.weight.data
                elif 'mlp' in component:
                    if hasattr(layer, 'intermediate'):
                        return layer.intermediate.dense.weight.data
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get component weight: {str(e)}")
            return None
    
    def analyze_attention_similarity(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module,
        tokenizer: Any,
        test_samples: List[Tuple[str, str]],
        model_name: str,
        num_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze similarity of attention patterns between original and pruned models
        
        Args:
            original_model: Original model
            pruned_model: Pruned model
            tokenizer: Tokenizer
            test_samples: Test query-document pairs
            model_name: Model name
            num_samples: Number of samples to analyze
            
        Returns:
            Attention similarity metrics
        """
        try:
            logger.info(f"Analyzing attention similarity for {model_name}")
            
            original_model.eval()
            pruned_model.eval()
            original_model.to(self.device)
            pruned_model.to(self.device)
            
            attention_similarities = defaultdict(list)
            layer_similarities = []
            
            # Sample test data
            samples = test_samples[:min(num_samples, len(test_samples))]
            
            for query, document in tqdm(samples, desc="Computing attention similarity"):
                # Tokenize input
                inputs = tokenizer(
                    query, document,
                    padding=True, truncation=True,
                    max_length=256, return_tensors='pt'
                ).to(self.device)
                
                # Get attention weights from both models
                with torch.no_grad():
                    # Original model
                    orig_outputs = original_model.bert(
                        **inputs,
                        output_attentions=True
                    )
                    orig_attentions = orig_outputs.attentions
                    
                    # Pruned model
                    pruned_outputs = pruned_model.bert(
                        **inputs,
                        output_attentions=True
                    )
                    pruned_attentions = pruned_outputs.attentions
                
                # Compare attention patterns at each layer
                for layer_idx in range(min(len(orig_attentions), len(pruned_attentions))):
                    orig_attn = orig_attentions[layer_idx]
                    pruned_attn = pruned_attentions[layer_idx]
                    
                    # Compute cosine similarity
                    orig_flat = orig_attn.flatten()
                    pruned_flat = pruned_attn.flatten()
                    
                    similarity = F.cosine_similarity(orig_flat, pruned_flat, dim=0).item()
                    attention_similarities[f'layer_{layer_idx}'].append(similarity)
                    
                    # Also compute correlation
                    orig_np = orig_flat.cpu().numpy()
                    pruned_np = pruned_flat.cpu().numpy()
                    
                    if len(orig_np) > 1:
                        correlation = np.corrcoef(orig_np, pruned_np)[0, 1]
                        attention_similarities[f'layer_{layer_idx}_corr'].append(correlation)
            
            # Aggregate results
            results = {
                'model': model_name,
                'num_samples': len(samples),
                'layer_similarities': {}
            }
            
            for layer_key, similarities in attention_similarities.items():
                if similarities:
                    results['layer_similarities'][layer_key] = {
                        'mean': np.mean(similarities),
                        'std': np.std(similarities),
                        'min': np.min(similarities),
                        'max': np.max(similarities)
                    }
            
            # Calculate overall similarity
            all_similarities = []
            for key, sims in attention_similarities.items():
                if 'corr' not in key:
                    all_similarities.extend(sims)
            
            if all_similarities:
                results['overall_similarity'] = {
                    'mean': np.mean(all_similarities),
                    'std': np.std(all_similarities),
                    'min': np.min(all_similarities),
                    'max': np.max(all_similarities)
                }
                
                logger.info(f"Overall attention similarity: {results['overall_similarity']['mean']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Attention similarity analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def analyze_importance_correlation(
        self,
        pruned_model: nn.Module,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Analyze correlation between original importance scores and pruned weights
        
        Args:
            pruned_model: Pruned model
            model_name: Model name
            
        Returns:
            Importance correlation metrics
        """
        try:
            logger.info(f"Analyzing importance correlation for {model_name}")
            
            original_scores = []
            remaining_weights = []
            component_names = []
            
            # Get remaining weight magnitudes for each component
            for component_name, importance in self.original_importance.items():
                # Parse component name
                parts = component_name.split('_')
                if len(parts) >= 3 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    comp_type = parts[2]
                    
                    # Get weight magnitude
                    weight = self._get_component_weight(pruned_model, layer_idx, comp_type)
                    
                    if weight is not None:
                        weight_magnitude = torch.norm(weight).item()
                        
                        original_scores.append(importance)
                        remaining_weights.append(weight_magnitude)
                        component_names.append(component_name)
            
            # Calculate correlations
            results = {
                'model': model_name,
                'num_components': len(original_scores)
            }
            
            if len(original_scores) > 1:
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(original_scores, remaining_weights)
                results['pearson_correlation'] = pearson_r
                results['pearson_p_value'] = pearson_p
                
                # Spearman correlation
                spearman_r, spearman_p = spearmanr(original_scores, remaining_weights)
                results['spearman_correlation'] = spearman_r
                results['spearman_p_value'] = spearman_p
                
                # Check if high-importance components are preserved
                threshold = np.percentile(original_scores, 75)  # Top 25%
                high_importance_mask = np.array(original_scores) > threshold
                
                if np.any(high_importance_mask):
                    high_imp_weights = np.array(remaining_weights)[high_importance_mask]
                    low_imp_weights = np.array(remaining_weights)[~high_importance_mask]
                    
                    results['high_importance_avg_weight'] = np.mean(high_imp_weights)
                    results['low_importance_avg_weight'] = np.mean(low_imp_weights)
                    results['importance_weight_ratio'] = (results['high_importance_avg_weight'] / 
                                                         (results['low_importance_avg_weight'] + 1e-8))
                
                logger.info(f"Importance-weight correlation: {pearson_r:.4f} (p={pearson_p:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Importance correlation analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def analyze_activation_patterns(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module,
        tokenizer: Any,
        test_samples: List[Tuple[str, str]],
        model_name: str,
        num_samples: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze similarity of activation patterns
        
        Args:
            original_model: Original model
            pruned_model: Pruned model
            tokenizer: Tokenizer
            test_samples: Test samples
            model_name: Model name
            num_samples: Number of samples to analyze
            
        Returns:
            Activation pattern metrics
        """
        try:
            logger.info(f"Analyzing activation patterns for {model_name}")
            
            original_model.eval()
            pruned_model.eval()
            original_
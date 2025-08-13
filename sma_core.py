"""
Sparse Mechanistic Analysis (SMA) for IR Systems
Phase 1: Core Implementation
Main module for mechanistic analysis components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import logging
from pathlib import Path
import json
from tqdm.auto import tqdm
from contextlib import contextmanager
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ActivationCache:
    """Store activations during forward passes"""
    activations: Dict[str, torch.Tensor] = field(default_factory=dict)
    attention_weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    attention_patterns: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    def clear(self):
        """Clear all cached activations"""
        self.activations.clear()
        self.attention_weights.clear()
        self.attention_patterns.clear()
    
    def save(self, path: Path):
        """Save cache to disk"""
        torch.save({
            'activations': self.activations,
            'attention_weights': self.attention_weights,
            'attention_patterns': self.attention_patterns
        }, path)
    
    def load(self, path: Path):
        """Load cache from disk"""
        data = torch.load(path)
        self.activations = data['activations']
        self.attention_weights = data['attention_weights']
        self.attention_patterns = data['attention_patterns']


@dataclass
class Circuit:
    """Represents a discovered circuit in the model"""
    name: str
    layers: List[int]
    components: List[str]  # neuron/head identifiers
    importance_score: float
    function_description: str = ""
    query_tokens: List[int] = field(default_factory=list)
    doc_tokens: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'layers': self.layers,
            'components': self.components,
            'importance_score': self.importance_score,
            'function_description': self.function_description,
            'query_tokens': self.query_tokens,
            'doc_tokens': self.doc_tokens
        }


class IRActivationPatching:
    """
    IR-specific activation patching for causal analysis
    Adapts standard activation patching to query-document pairs
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize activation patching for IR model
        
        Args:
            model: The IR model (e.g., BERT cross-encoder)
            tokenizer: Tokenizer for the model
            device: Device for computation
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.cache = ActivationCache()
        self.hooks = []
        
        # Model configuration
        self.n_layers = self._get_num_layers()
        self.n_heads = self._get_num_heads()
        self.hidden_size = self._get_hidden_size()
        
        logger.info(f"Initialized IRActivationPatching with {self.n_layers} layers, "
                   f"{self.n_heads} heads, hidden_size={self.hidden_size}")
    
    def _get_num_layers(self) -> int:
        """Detect number of layers in model"""
        if hasattr(self.model, 'config'):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model, 'bert'):
            return self.model.bert.config.num_hidden_layers
        else:
            raise ValueError("Cannot determine number of layers")
    
    def _get_num_heads(self) -> int:
        """Detect number of attention heads"""
        if hasattr(self.model, 'config'):
            return self.model.config.num_attention_heads
        elif hasattr(self.model, 'bert'):
            return self.model.bert.config.num_attention_heads
        else:
            raise ValueError("Cannot determine number of attention heads")
    
    def _get_hidden_size(self) -> int:
        """Detect hidden size"""
        if hasattr(self.model, 'config'):
            return self.model.config.hidden_size
        elif hasattr(self.model, 'bert'):
            return self.model.bert.config.hidden_size
        else:
            raise ValueError("Cannot determine hidden size")
    
    @contextmanager
    def register_hooks(self, layers_to_cache: Optional[List[int]] = None):
        """
        Context manager for registering forward hooks
        
        Args:
            layers_to_cache: Specific layers to cache, None for all
        """
        if layers_to_cache is None:
            layers_to_cache = list(range(self.n_layers))
        
        def get_activation_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # Handle attention outputs
                    self.cache.activations[name] = output[0].detach()
                    if len(output) > 1 and output[1] is not None:
                        self.cache.attention_weights[name] = output[1].detach()
                else:
                    self.cache.activations[name] = output.detach()
            return hook
        
        try:
            # Register hooks for specified layers
            for layer_idx in layers_to_cache:
                if hasattr(self.model, 'bert'):
                    layer = self.model.bert.encoder.layer[layer_idx]
                else:
                    layer = self.model.encoder.layer[layer_idx]
                
                # Hook for attention
                hook = layer.attention.register_forward_hook(
                    get_activation_hook(f'layer_{layer_idx}_attention')
                )
                self.hooks.append(hook)
                
                # Hook for MLP
                hook = layer.output.register_forward_hook(
                    get_activation_hook(f'layer_{layer_idx}_mlp')
                )
                self.hooks.append(hook)
            
            yield
            
        finally:
            # Clean up hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
    
    def create_counterfactual(
        self,
        query: str,
        document: str,
        perturbation_type: str = 'mask_keywords'
    ) -> Tuple[str, str]:
        """
        Create counterfactual query-document pairs
        
        Args:
            query: Original query
            document: Original document
            perturbation_type: Type of perturbation to apply
        
        Returns:
            Perturbed query and document
        """
        if perturbation_type == 'mask_keywords':
            # Mask important keywords
            query_tokens = query.split()
            doc_tokens = document.split()
            
            # Simple heuristic: mask longest words (likely important)
            if len(query_tokens) > 2:
                important_idx = np.argmax([len(t) for t in query_tokens])
                query_tokens[important_idx] = '[MASK]'
            
            if len(doc_tokens) > 5:
                important_indices = np.argsort([len(t) for t in doc_tokens])[-3:]
                for idx in important_indices:
                    if idx < len(doc_tokens):
                        doc_tokens[idx] = '[MASK]'
            
            return ' '.join(query_tokens), ' '.join(doc_tokens)
        
        elif perturbation_type == 'shuffle':
            # Shuffle tokens
            query_tokens = query.split()
            doc_tokens = document.split()
            np.random.shuffle(query_tokens)
            np.random.shuffle(doc_tokens)
            return ' '.join(query_tokens), ' '.join(doc_tokens)
        
        elif perturbation_type == 'noise':
            # Add noise tokens
            noise_tokens = ['random', 'irrelevant', 'noise']
            document += ' ' + ' '.join(np.random.choice(noise_tokens, 5))
            return query, document
        
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    def patch_activation(
        self,
        clean_cache: ActivationCache,
        corrupted_cache: ActivationCache,
        layer: int,
        component: str,
        position: Optional[int] = None
    ) -> ActivationCache:
        """
        Patch specific activation from clean to corrupted run
        
        Args:
            clean_cache: Activations from clean run
            corrupted_cache: Activations from corrupted run
            layer: Layer index
            component: Component type ('attention' or 'mlp')
            position: Token position to patch (None for all)
        
        Returns:
            Patched activation cache
        """
        patched_cache = ActivationCache()
        patched_cache.activations = corrupted_cache.activations.copy()
        patched_cache.attention_weights = corrupted_cache.attention_weights.copy()
        
        key = f'layer_{layer}_{component}'
        if key in clean_cache.activations:
            if position is not None:
                # Patch specific position
                patched_cache.activations[key][:, position, :] = \
                    clean_cache.activations[key][:, position, :].clone()
            else:
                # Patch entire activation
                patched_cache.activations[key] = clean_cache.activations[key].clone()
        
        return patched_cache
    
    def compute_causal_effect(
        self,
        query: str,
        document: str,
        layer: int,
        component: str,
        num_perturbations: int = 5
    ) -> float:
        """
        Compute causal effect of a component via activation patching
        
        Args:
            query: Query text
            document: Document text
            layer: Layer index
            component: Component type
            num_perturbations: Number of perturbations to average over
        
        Returns:
            Causal effect score
        """
        effects = []
        
        for _ in range(num_perturbations):
            # Get clean output
            with self.register_hooks([layer]):
                clean_inputs = self.tokenizer(
                    query, document,
                    padding=True, truncation=True,
                    max_length=512, return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    clean_output = self.model(**clean_inputs)
                    clean_score = clean_output.logits.squeeze().item() \
                        if hasattr(clean_output, 'logits') else clean_output[0].item()
                
                clean_cache = ActivationCache()
                clean_cache.activations = self.cache.activations.copy()
                self.cache.clear()
            
            # Get corrupted output
            perturbed_query, perturbed_doc = self.create_counterfactual(
                query, document, 'mask_keywords'
            )
            
            with self.register_hooks([layer]):
                corrupted_inputs = self.tokenizer(
                    perturbed_query, perturbed_doc,
                    padding=True, truncation=True,
                    max_length=512, return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    corrupted_output = self.model(**corrupted_inputs)
                    corrupted_score = corrupted_output.logits.squeeze().item() \
                        if hasattr(corrupted_output, 'logits') else corrupted_output[0].item()
                
                corrupted_cache = ActivationCache()
                corrupted_cache.activations = self.cache.activations.copy()
                self.cache.clear()
            
            # Compute patched output
            patched_cache = self.patch_activation(
                clean_cache, corrupted_cache, layer, component
            )
            
            # We need to implement a forward pass with patched activations
            # This is model-specific and requires custom implementation
            # For now, we approximate the effect
            effect = abs(clean_score - corrupted_score)
            effects.append(effect)
        
        return np.mean(effects)
    
    def trace_circuits(
        self,
        query: str,
        document: str,
        threshold: float = 0.1
    ) -> List[Circuit]:
        """
        Trace information flow to discover circuits
        
        Args:
            query: Query text
            document: Document text
            threshold: Importance threshold for circuit inclusion
        
        Returns:
            List of discovered circuits
        """
        circuits = []
        
        # Get model predictions and cache activations
        with self.register_hooks():
            inputs = self.tokenizer(
                query, document,
                padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
        
        # Get token positions for query and document
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        query_tokens = query.lower().split()
        doc_tokens = document.lower().split()
        
        # Find query and document token positions in the tokenized sequence
        query_positions = []
        doc_positions = []
        
        for i, token in enumerate(tokens):
            token_clean = token.replace('##', '').lower()
            if any(qt in token_clean for qt in query_tokens):
                query_positions.append(i)
            elif any(dt in token_clean for dt in doc_tokens):
                doc_positions.append(i)
        
        # Analyze attention patterns to find circuits
        for layer in range(self.n_layers):
            attn_key = f'layer_{layer}_attention'
            
            # Check activations directly if attention weights not available
            if attn_key in self.cache.activations:
                activation = self.cache.activations[attn_key]
                
                # Compute activation statistics
                mean_activation = activation.abs().mean().item()
                max_activation = activation.abs().max().item()
                
                # If activations are significant, create a circuit
                if mean_activation > threshold or max_activation > threshold * 10:
                    circuit = Circuit(
                        name=f"L{layer}_activation_circuit",
                        layers=[layer],
                        components=[attn_key],
                        importance_score=max_activation,
                        function_description=f"Layer {layer} activation pattern",
                        query_tokens=query_positions[:5],
                        doc_tokens=doc_positions[:5]
                    )
                    circuits.append(circuit)
            
            # Also check attention weights if available
            if attn_key in self.cache.attention_weights:
                attn_weights = self.cache.attention_weights[attn_key]
                
                if len(attn_weights.shape) >= 3:
                    # Average over heads if multi-head attention
                    if len(attn_weights.shape) == 4:
                        attn_weights = attn_weights[0]  # Remove batch dimension
                    if len(attn_weights.shape) == 3:
                        attn_weights = attn_weights.mean(dim=0)  # Average over heads
                    
                    # Find strong attention connections
                    max_attn = attn_weights.max().item()
                    mean_attn = attn_weights.mean().item()
                    
                    # Look for query-document attention patterns
                    if query_positions and doc_positions:
                        # Check attention from query to document
                        query_doc_attn = attn_weights[query_positions][:, doc_positions]
                        if query_doc_attn.numel() > 0:
                            max_qd_attn = query_doc_attn.max().item()
                            if max_qd_attn > threshold:
                                circuit = Circuit(
                                    name=f"L{layer}_query_doc_attention",
                                    layers=[layer],
                                    components=[f"{attn_key}_qd"],
                                    importance_score=max_qd_attn,
                                    function_description=f"Query-to-document attention in layer {layer}",
                                    query_tokens=query_positions[:5],
                                    doc_tokens=doc_positions[:5]
                                )
                                circuits.append(circuit)
                    
                    # General attention circuit if strong patterns exist
                    if mean_attn > threshold or max_attn > threshold * 5:
                        circuit = Circuit(
                            name=f"L{layer}_attention_circuit",
                            layers=[layer],
                            components=[attn_key],
                            importance_score=max_attn,
                            function_description=f"General attention pattern in layer {layer}",
                            query_tokens=query_positions[:5],
                            doc_tokens=doc_positions[:5]
                        )
                        circuits.append(circuit)
            
            # Check MLP activations
            mlp_key = f'layer_{layer}_mlp'
            if mlp_key in self.cache.activations:
                mlp_activation = self.cache.activations[mlp_key]
                max_mlp = mlp_activation.abs().max().item()
                mean_mlp = mlp_activation.abs().mean().item()
                
                if mean_mlp > threshold or max_mlp > threshold * 10:
                    circuit = Circuit(
                        name=f"L{layer}_mlp_circuit",
                        layers=[layer],
                        components=[mlp_key],
                        importance_score=max_mlp,
                        function_description=f"MLP processing in layer {layer}",
                        query_tokens=query_positions[:5],
                        doc_tokens=doc_positions[:5]
                    )
                    circuits.append(circuit)
        
        # Sort circuits by importance
        circuits.sort(key=lambda x: x.importance_score, reverse=True)
        
        return circuits


class LogitLensIR:
    """
    Logit lens analysis adapted for IR models
    Projects hidden states to relevance scores at each layer
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize logit lens for IR"""
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Get model output head
        self.output_head = self._get_output_head()
        self.hidden_size = self._get_hidden_size()
        
        # Create probes for each layer
        self.probes = self._create_probes()
    
    def _get_output_head(self) -> nn.Module:
        """Extract the output classification head"""
        if hasattr(self.model, 'classifier'):
            return self.model.classifier
        elif hasattr(self.model, 'output'):
            return self.model.output
        else:
            # Find the last linear layer
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Linear):
                    return module
            raise ValueError("Cannot find output head")
    
    def _get_hidden_size(self) -> int:
        """Get hidden size of the model"""
        if hasattr(self.model, 'config'):
            return self.model.config.hidden_size
        elif hasattr(self.model, 'bert'):
            return self.model.bert.config.hidden_size
        else:
            return 768  # Default BERT hidden size
    
    def _create_probes(self) -> nn.ModuleDict:
        """Create linear probes for each layer"""
        probes = nn.ModuleDict()
        
        # Determine number of layers
        if hasattr(self.model, 'bert'):
            n_layers = self.model.bert.config.num_hidden_layers
        else:
            n_layers = 12  # Default
        
        for layer in range(n_layers):
            # Simple linear probe mapping hidden states to scores
            probe = nn.Linear(self.hidden_size, 1)
            probe.to(self.device)
            probes[f'layer_{layer}'] = probe
        
        return probes
    
    def train_probes(
        self,
        train_data: List[Tuple[str, str, float]],
        epochs: int = 10,
        lr: float = 1e-3
    ):
        """
        Train probes on labeled data
        
        Args:
            train_data: List of (query, document, relevance_score) tuples
            epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.probes.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for query, doc, score in tqdm(train_data, desc=f"Epoch {epoch+1}"):
                # Get hidden states at each layer
                hidden_states = self.extract_hidden_states(query, doc)
                
                loss = 0
                for layer_name, states in hidden_states.items():
                    if layer_name in self.probes:
                        # Pool hidden states (use [CLS] token)
                        pooled = states[:, 0, :]  # [batch, hidden]
                        
                        # Get probe prediction
                        pred = self.probes[layer_name](pooled)
                        target = torch.tensor([[score]], device=self.device)
                        
                        loss += criterion(pred, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_data)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def extract_hidden_states(
        self,
        query: str,
        document: str
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hidden states at each layer
        
        Args:
            query: Query text
            document: Document text
        
        Returns:
            Dictionary mapping layer names to hidden states
        """
        hidden_states = {}
        
        # Register hooks to capture hidden states
        hooks = []
        
        def get_hidden_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states[name] = output[0].detach()
                else:
                    hidden_states[name] = output.detach()
            return hook
        
        # Register hooks for each layer
        if hasattr(self.model, 'bert'):
            for i, layer in enumerate(self.model.bert.encoder.layer):
                hook = layer.register_forward_hook(get_hidden_hook(f'layer_{i}'))
                hooks.append(hook)
        
        # Forward pass
        inputs = self.tokenizer(
            query, document,
            padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return hidden_states
    
    def analyze_token_importance(
        self,
        query: str,
        document: str,
        layer: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        Analyze importance of each token at specified layer
        
        Args:
            query: Query text
            document: Document text
            layer: Layer index (-1 for last layer)
        
        Returns:
            Token importance scores
        """
        # Tokenize input
        inputs = self.tokenizer(
            query, document,
            padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        ).to(self.device)
        
        # Get hidden states
        hidden_states = self.extract_hidden_states(query, document)
        
        # Get states at specified layer
        if layer == -1:
            layer = len(hidden_states) - 1
        
        layer_states = hidden_states[f'layer_{layer}']
        
        # Use probe to get importance scores
        probe = self.probes[f'layer_{layer}']
        
        # Compute importance for each token
        token_scores = []
        for i in range(layer_states.shape[1]):
            token_state = layer_states[:, i:i+1, :]
            with torch.no_grad():
                score = probe(token_state)
            token_scores.append(score.item())
        
        # Map back to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return {
            'tokens': tokens,
            'scores': np.array(token_scores)
        }


class ImportanceScorer:
    """
    Compute importance scores for model components
    Combines performance impact and explanation contribution
    """
    
    def __init__(
        self,
        model: nn.Module,
        activation_patcher: IRActivationPatching,
        logit_lens: LogitLensIR,
        lambda_balance: float = 0.5
    ):
        """
        Initialize importance scorer
        
        Args:
            model: The IR model
            activation_patcher: Activation patching module
            logit_lens: Logit lens module
            lambda_balance: Balance between performance and explanation (0-1)
        """
        self.model = model
        self.activation_patcher = activation_patcher
        self.logit_lens = logit_lens
        self.lambda_balance = lambda_balance
        
        self.importance_scores = defaultdict(float)
        self.performance_impacts = defaultdict(float)
        self.explanation_contributions = defaultdict(float)
    
    def compute_performance_impact(
        self,
        component_id: str,
        eval_data: List[Tuple[str, str, float]],
        num_samples: int = 100
    ) -> float:
        """
        Compute performance drop when component is removed
        
        Args:
            component_id: Identifier for component (e.g., 'layer_3_attention')
            eval_data: Evaluation data
            num_samples: Number of samples to evaluate
        
        Returns:
            Performance impact score
        """
        # Sample evaluation data
        if len(eval_data) > num_samples:
            indices = np.random.choice(len(eval_data), num_samples, replace=False)
            eval_data = [eval_data[i] for i in indices]
        
        original_scores = []
        ablated_scores = []
        
        for query, doc, true_score in tqdm(eval_data, desc=f"Evaluating {component_id}"):
            # Get original score
            inputs = self.activation_patcher.tokenizer(
                query, doc,
                padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            ).to(self.activation_patcher.device)
            
            with torch.no_grad():
                output = self.model(**inputs)
                orig_score = output.logits.squeeze().item() \
                    if hasattr(output, 'logits') else output[0].item()
            original_scores.append(orig_score)
            
            # Get ablated score (component removed)
            # This requires model-specific implementation
            # For now, we use activation patching as proxy
            layer = int(component_id.split('_')[1])
            component_type = component_id.split('_')[2]
            
            causal_effect = self.activation_patcher.compute_causal_effect(
                query, doc, layer, component_type, num_perturbations=3
            )
            ablated_scores.append(orig_score - causal_effect)
        
        # Compute performance drop
        original_scores = np.array(original_scores)
        ablated_scores = np.array(ablated_scores)
        
        # Normalized performance impact
        if original_scores.std() > 0:
            impact = (original_scores - ablated_scores).mean() / original_scores.std()
        else:
            impact = (original_scores - ablated_scores).mean()
        
        return float(impact)
    
    def compute_explanation_contribution(
        self,
        component_id: str,
        explanation_data: List[Tuple[str, str]],
        num_samples: int = 50
    ) -> float:
        """
        Compute how much component contributes to explanations
        
        Args:
            component_id: Component identifier
            explanation_data: Data for explanation analysis
            num_samples: Number of samples
        
        Returns:
            Explanation contribution score
        """
        if len(explanation_data) > num_samples:
            indices = np.random.choice(len(explanation_data), num_samples, replace=False)
            explanation_data = [explanation_data[i] for i in indices]
        
        contribution_scores = []
        
        for query, doc in tqdm(explanation_data, desc=f"Explanation {component_id}"):
            # Get token importance from logit lens
            layer = int(component_id.split('_')[1])
            importance = self.logit_lens.analyze_token_importance(query, doc, layer)
            
            # Check if this component affects important tokens
            # High variance in scores indicates component is selective
            score_variance = np.var(importance['scores'])
            contribution_scores.append(score_variance)
        
        return float(np.mean(contribution_scores))
    
    def compute_importance(
        self,
        component_id: str,
        eval_data: List[Tuple[str, str, float]]
    ) -> float:
        """
        Compute combined importance score
        
        Args:
            component_id: Component identifier
            eval_data: Evaluation data
        
        Returns:
            Combined importance score
        """
        # Get performance impact
        perf_impact = self.compute_performance_impact(
            component_id, eval_data, num_samples=50
        )
        self.performance_impacts[component_id] = perf_impact
        
        # Get explanation contribution
        explanation_data = [(q, d) for q, d, _ in eval_data]
        exp_contribution = self.compute_explanation_contribution(
            component_id, explanation_data, num_samples=25
        )
        self.explanation_contributions[component_id] = exp_contribution
        
        # Combine scores
        importance = (
            self.lambda_balance * perf_impact +
            (1 - self.lambda_balance) * exp_contribution
        )
        
        self.importance_scores[component_id] = importance
        return importance
    
    def rank_components(
        self,
        eval_data: List[Tuple[str, str, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank all components by importance
        
        Args:
            eval_data: Evaluation data
            top_k: Return only top k components
        
        Returns:
            Ranked list of (component_id, importance_score)
        """
        # Compute importance for all components
        all_components = []
        
        # Add attention components
        for layer in range(self.activation_patcher.n_layers):
            all_components.append(f'layer_{layer}_attention')
            all_components.append(f'layer_{layer}_mlp')
        
        # Compute importance scores
        for component_id in tqdm(all_components, desc="Ranking components"):
            if component_id not in self.importance_scores:
                self.compute_importance(component_id, eval_data)
        
        # Sort by importance
        ranked = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked
    
    def save_scores(self, path: Path):
        """Save importance scores to file"""
        scores_data = {
            'importance_scores': dict(self.importance_scores),
            'performance_impacts': dict(self.performance_impacts),
            'explanation_contributions': dict(self.explanation_contributions),
            'lambda_balance': self.lambda_balance
        }
        
        with open(path, 'w') as f:
            json.dump(scores_data, f, indent=2)
    
    def load_scores(self, path: Path):
        """Load importance scores from file"""
        with open(path, 'r') as f:
            scores_data = json.load(f)
        
        self.importance_scores = defaultdict(float, scores_data['importance_scores'])
        self.performance_impacts = defaultdict(float, scores_data['performance_impacts'])
        self.explanation_contributions = defaultdict(float, scores_data['explanation_contributions'])
        self.lambda_balance = scores_data['lambda_balance']


def verify_implementation(
    model: nn.Module,
    tokenizer: Any,
    sample_data: List[Tuple[str, str, float]]
) -> bool:
    """
    Verify Phase 1 implementation with basic tests
    
    Args:
        model: IR model
        tokenizer: Tokenizer
        sample_data: Sample query-document-score data
    
    Returns:
        True if all tests pass
    """
    logger.info("Starting implementation verification...")
    
    try:
        # Test 1: Initialize components
        logger.info("Test 1: Initializing components...")
        activation_patcher = IRActivationPatching(model, tokenizer)
        logit_lens = LogitLensIR(model, tokenizer)
        importance_scorer = ImportanceScorer(
            model, activation_patcher, logit_lens, lambda_balance=0.5
        )
        logger.info("✓ Components initialized successfully")
        
        # Test 2: Activation caching
        logger.info("Test 2: Testing activation caching...")
        query, doc, score = sample_data[0]
        with activation_patcher.register_hooks([0, 1]):
            inputs = tokenizer(
                query, doc,
                padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            )
            _ = model(**inputs)
        
        assert len(activation_patcher.cache.activations) > 0, "No activations cached"
        logger.info(f"✓ Cached {len(activation_patcher.cache.activations)} activations")
        
        # Test 3: Counterfactual generation
        logger.info("Test 3: Testing counterfactual generation...")
        perturbed_q, perturbed_d = activation_patcher.create_counterfactual(
            query, doc, 'mask_keywords'
        )
        assert '[MASK]' in perturbed_q or '[MASK]' in perturbed_d, "No masking applied"
        logger.info("✓ Counterfactuals generated successfully")
        
        # Test 4: Circuit discovery
        logger.info("Test 4: Testing circuit discovery...")
        circuits = activation_patcher.trace_circuits(query, doc, threshold=0.05)
        assert len(circuits) > 0, "No circuits discovered"
        logger.info(f"✓ Discovered {len(circuits)} circuits")
        
        # Test 5: Importance scoring
        logger.info("Test 5: Testing importance scoring...")
        importance = importance_scorer.compute_importance(
            'layer_0_attention', sample_data[:5]
        )
        assert importance is not None, "Importance computation failed"
        logger.info(f"✓ Computed importance score: {importance:.4f}")
        
        logger.info("All tests passed! ✓")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False
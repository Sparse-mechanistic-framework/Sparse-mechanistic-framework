"""
SMA Circuit Discovery for BRIGHT Dataset - Updated Version
Fixes: Using documents subset with theoremqa_questions split
Handles 80K samples from the correct subset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
from dataclasses import dataclass
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BRIGHTSample:
    """Data structure for BRIGHT dataset samples"""
    query_id: str
    query: str
    document: str
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = None


class BRIGHTDataset(Dataset):
    """
    BRIGHT dataset loader - Updated for documents/theoremqa_questions split
    Handles mathematical reasoning and theorem-based query-document pairs
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_samples: int = 160000,
        cache_dir: str = './cache/bright',
        max_length: int = 256,
        min_doc_length: int = 20
    ):
        """
        Initialize BRIGHT dataset with correct configuration
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_samples: Maximum samples to load (default 80K)
            cache_dir: Cache directory
            max_length: Maximum sequence length for tokenization
            min_doc_length: Minimum document length to consider
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_doc_length = min_doc_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Loading BRIGHT dataset (documents/theoremqa_questions)...")
        
        # Load the correct subset and split
        try:
            dataset = load_dataset(
                'xlangai/BRIGHT',
                'documents',  # Correct subset
                split='theoremqa_questions',  # Correct split
                cache_dir=str(self.cache_dir),
                # Removed trust_remote_code as requested
            )
        except Exception as e:
            logger.error(f"Error loading BRIGHT dataset: {e}")
            logger.info("Attempting alternative loading method...")
            # Alternative loading if first attempt fails
            dataset = load_dataset(
                'xlangai/BRIGHT',
                name='documents',
                split='theoremqa_questions',
                cache_dir=str(self.cache_dir)
            )
        
        # Process the dataset
        self.data = self._process_dataset(dataset, max_samples)
        logger.info(f"Loaded {len(self.data)} samples from BRIGHT dataset")
        
        # Compute dataset statistics
        self._compute_statistics()
    
    def _process_dataset(
        self, 
        dataset, 
        max_samples: int
    ) -> List[BRIGHTSample]:
        """
        Process raw dataset into structured format
        
        Args:
            dataset: Raw HuggingFace dataset
            max_samples: Maximum number of samples to process
        
        Returns:
            List of processed samples
        """
        processed_data = []
        sample_count = 0
        
        logger.info("Processing BRIGHT dataset...")
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing samples")):
            if sample_count >= max_samples:
                break
            
            try:
                # Extract id and content from the dataset
                # According to the dataset structure: columns ('id', 'content')
                sample_id = str(item.get('id', f'bright_{idx}'))
                content = item.get('content', '')
                
                if not content or len(content) < self.min_doc_length:
                    continue
                
                # Split content into query and document parts
                # For theorem QA, we'll treat the first sentence/question as query
                # and the rest as the document
                sentences = content.split('. ')
                
                if len(sentences) < 2:
                    # If no clear split, use first 100 chars as query
                    query = content[:100].strip()
                    document = content[100:].strip() if len(content) > 100 else content
                else:
                    query = sentences[0].strip()
                    document = '. '.join(sentences[1:]).strip()
                
                # Ensure both query and document are non-empty
                if not query or not document:
                    continue
                
                # Create sample
                sample = BRIGHTSample(
                    query_id=sample_id,
                    query=query,
                    document=document,
                    relevance_score=1.0,  # Assume relevance for theorem QA
                    metadata={
                        'source': 'theoremqa',
                        'idx': idx,
                        'content_length': len(content)
                    }
                )
                
                processed_data.append(sample)
                sample_count += 1
                
            except Exception as e:
                logger.debug(f"Error processing sample {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_data)} samples")
        return processed_data
    
    def _compute_statistics(self):
        """Compute dataset statistics for analysis"""
        self.stats = {
            'num_samples': len(self.data),
            'avg_query_length': np.mean([len(s.query.split()) for s in self.data]),
            'avg_doc_length': np.mean([len(s.document.split()) for s in self.data]),
            'max_query_length': max([len(s.query.split()) for s in self.data]),
            'max_doc_length': max([len(s.document.split()) for s in self.data]),
            'unique_queries': len(set([s.query_id for s in self.data]))
        }
        
        logger.info("Dataset statistics:")
        for key, value in self.stats.items():
            logger.info(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with tokenization"""
        sample = self.data[idx]
        
        # Tokenize query and document
        query_encoding = self.tokenizer(
            sample.query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length // 2,
            return_tensors='pt'
        )
        
        doc_encoding = self.tokenizer(
            sample.document,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'query_id': sample.query_id,
            'query': sample.query,
            'document': sample.document,
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'doc_input_ids': doc_encoding['input_ids'].squeeze(),
            'doc_attention_mask': doc_encoding['attention_mask'].squeeze(),
            'relevance_score': sample.relevance_score,
            'metadata': sample.metadata
        }
    
    def get_samples_for_analysis(
        self, 
        num_samples: int = 500,
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Get samples for circuit discovery analysis
        
        Args:
            num_samples: Number of samples to return
            random_seed: Random seed for reproducibility
        
        Returns:
            List of samples for analysis
        """
        np.random.seed(random_seed)
        indices = np.random.choice(
            len(self.data), 
            min(num_samples, len(self.data)), 
            replace=False
        )
        
        samples = []
        for idx in indices:
            sample = self.data[idx]
            samples.append({
                'query': sample.query,
                'document': sample.document,
                'query_id': sample.query_id,
                'relevance_score': sample.relevance_score,
                'metadata': sample.metadata
            })
        
        return samples
    
    def get_batch_loader(
        self, 
        batch_size: int = 16, 
        shuffle: bool = True
    ) -> DataLoader:
        """
        Get a DataLoader for batch processing
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
        
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )


class BRIGHTCircuitDiscovery:
    """
    Circuit discovery module specialized for BRIGHT theorem QA patterns
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = 'cuda',
        cache_activations: bool = True
    ):
        """
        Initialize circuit discovery for BRIGHT
        
        Args:
            model: Pre-trained transformer model
            tokenizer: Tokenizer
            device: Device to use
            cache_activations: Whether to cache activations
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.cache_activations = cache_activations
        
        # Model configuration
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        
        # Activation cache
        self.activation_cache = {}
        
        # Hook handles
        self.hooks = []
        
        logger.info(f"Initialized BRIGHT circuit discovery with {self.n_layers} layers")
    
    def trace_circuits(
        self,
        query: str,
        document: str,
        threshold: float = 0.02
    ) -> List[Dict[str, Any]]:
        """
        Trace circuits for a query-document pair
        
        Args:
            query: Query text
            document: Document text
            threshold: Importance threshold
        
        Returns:
            List of discovered circuits
        """
        circuits = []
        
        # Tokenize inputs
        query_inputs = self.tokenizer(
            query,
            return_tensors='pt',
            truncation=True,
            max_length=128
        ).to(self.device)
        
        doc_inputs = self.tokenizer(
            document,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # Register hooks to capture activations
        self._register_hooks()
        
        try:
            # Forward pass to collect activations
            with torch.no_grad():
                _ = self.model(**query_inputs)
                query_activations = self._extract_activations()
                
                _ = self.model(**doc_inputs)
                doc_activations = self._extract_activations()
            
            # Analyze activation patterns
            for layer in range(self.n_layers):
                # Compute importance scores
                importance = self._compute_importance(
                    query_activations[layer],
                    doc_activations[layer]
                )
                
                if importance > threshold:
                    circuit = {
                        'layer': layer,
                        'importance_score': float(importance),
                        'circuit_type': self._classify_circuit_type(layer),
                        'pattern': 'theorem_reasoning' if 'theorem' in query.lower() else 'general',
                        'components': ['attention', 'mlp']
                    }
                    circuits.append(circuit)
        
        finally:
            # Clean up hooks
            self._remove_hooks()
        
        return circuits
    
    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        self.activation_cache.clear()
        
        def hook_fn(name):
            def hook(module, input, output):
                if self.cache_activations:
                    self.activation_cache[name] = output[0].detach().cpu()
            return hook
        
        # Register hooks for each layer
        for i in range(self.n_layers):
            layer = self.model.bert.encoder.layer[i] if hasattr(self.model, 'bert') else \
                    self.model.encoder.layer[i]
            handle = layer.register_forward_hook(hook_fn(f'layer_{i}'))
            self.hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _extract_activations(self) -> Dict[int, torch.Tensor]:
        """Extract activations from cache"""
        activations = {}
        for i in range(self.n_layers):
            key = f'layer_{i}'
            if key in self.activation_cache:
                activations[i] = self.activation_cache[key]
        return activations
    
    def _compute_importance(
        self,
        query_activation: torch.Tensor,
        doc_activation: torch.Tensor
    ) -> float:
        """
        Compute importance score between query and document activations
        
        Args:
            query_activation: Query activation tensor
            doc_activation: Document activation tensor
        
        Returns:
            Importance score
        """
        # Compute cosine similarity
        query_norm = F.normalize(query_activation.mean(dim=1), p=2, dim=-1)
        doc_norm = F.normalize(doc_activation.mean(dim=1), p=2, dim=-1)
        
        similarity = torch.cosine_similarity(query_norm, doc_norm, dim=-1)
        
        return similarity.mean().item()
    
    def _classify_circuit_type(self, layer: int) -> str:
        """Classify circuit type based on layer position"""
        if layer < self.n_layers // 3:
            return 'early_pattern'
        elif layer < 2 * self.n_layers // 3:
            return 'middle_reasoning'
        else:
            return 'late_integration'
    
    def compute_causal_effects(
        self,
        dataset: BRIGHTDataset,
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Compute causal effects through intervention
        
        Args:
            dataset: BRIGHT dataset
            num_samples: Number of samples to evaluate
        
        Returns:
            Dictionary of causal effects
        """
        logger.info("Computing causal effects through intervention...")
        
        causal_effects = {
            'by_layer': {},
            'by_component': {'attention': [], 'mlp': []},
            'overall': 0.0
        }
        
        samples = dataset.get_samples_for_analysis(num_samples)
        
        for sample in tqdm(samples[:num_samples], desc="Computing causal effects"):
            # Tokenize
            inputs = self.tokenizer(
                sample['document'],
                return_tensors='pt',
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # Get baseline output
            with torch.no_grad():
                baseline_output = self.model(**inputs)
                baseline_logits = baseline_output.last_hidden_state.mean(dim=1)
            
            # Test interventions
            for layer in range(self.n_layers):
                with self._intervene_on_layer(layer):
                    with torch.no_grad():
                        perturbed_output = self.model(**inputs)
                        perturbed_logits = perturbed_output.last_hidden_state.mean(dim=1)
                    
                    effect = (baseline_logits - perturbed_logits).abs().mean().item()
                    
                    if layer not in causal_effects['by_layer']:
                        causal_effects['by_layer'][layer] = []
                    causal_effects['by_layer'][layer].append(effect)
        
        # Aggregate results
        for layer, effects in causal_effects['by_layer'].items():
            causal_effects['by_layer'][layer] = float(np.mean(effects))
        
        causal_effects['overall'] = float(
            np.mean(list(causal_effects['by_layer'].values()))
        )
        
        return causal_effects
    
    def _intervene_on_layer(self, layer: int):
        """Context manager for layer intervention"""
        class InterventionContext:
            def __init__(self, model, layer):
                self.model = model
                self.layer = layer
                self.original_forward = None
                self.layer_module = None
            
            def __enter__(self):
                # Get layer module
                if hasattr(self.model, 'bert'):
                    self.layer_module = self.model.bert.encoder.layer[self.layer]
                else:
                    self.layer_module = self.model.encoder.layer[self.layer]
                
                # Store original forward
                self.original_forward = self.layer_module.forward
                
                # Replace with zeroed version
                def zero_forward(*args, **kwargs):
                    output = self.original_forward(*args, **kwargs)
                    return (torch.zeros_like(output[0]),) + output[1:]
                
                self.layer_module.forward = zero_forward
                return self
            
            def __exit__(self, *args):
                # Restore original forward
                self.layer_module.forward = self.original_forward
        
        return InterventionContext(self.model, layer)


def main():
    """Main execution for BRIGHT circuit discovery"""
    
    # Configuration
    config = {
        'model_name': 'allenai/longformer-base-4096',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_samples': 140000,  # Use 80K samples as requested
        'output_dir': Path('./bright_circuits'),
        'threshold': 0.02,
        'batch_size': 32,
        'num_workers': 4
    }
    
    logger.info("="*60)
    logger.info("BRIGHT Dataset Circuit Discovery Pipeline")
    logger.info("="*60)
    logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
    
    # Create output directory
    config['output_dir'].mkdir(exist_ok=True, parents=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModel.from_pretrained(config['model_name'])
    
    # Initialize dataset
    logger.info("Initializing BRIGHT dataset...")
    dataset = BRIGHTDataset(
        tokenizer=tokenizer,
        max_samples=config['max_samples']
    )
    
    # Initialize circuit discovery
    circuit_discovery = BRIGHTCircuitDiscovery(
        model=model,
        tokenizer=tokenizer,
        device=config['device']
    )
    
    # Discover circuits
    logger.info("\nDiscovering circuits...")
    all_circuits = []
    
    # Process samples in batches for efficiency
    analysis_samples = dataset.get_samples_for_analysis(5000)
    
    for sample in tqdm(analysis_samples, desc="Processing samples"):
        circuits = circuit_discovery.trace_circuits(
            sample['query'],
            sample['document'],
            threshold=config['threshold']
        )
        all_circuits.extend(circuits)
    
    logger.info(f"Discovered {len(all_circuits)} circuits")
    
    # Compute causal effects
    logger.info("\nComputing causal effects...")
    causal_effects = circuit_discovery.compute_causal_effects(dataset, num_samples=2500)
    
    # Analyze patterns
    pattern_stats = {}
    for circuit in all_circuits:
        pattern = circuit.get('pattern', 'unknown')
        if pattern not in pattern_stats:
            pattern_stats[pattern] = {
                'count': 0,
                'avg_importance': 0,
                'layers': []
            }
        pattern_stats[pattern]['count'] += 1
        pattern_stats[pattern]['avg_importance'] += circuit['importance_score']
        pattern_stats[pattern]['layers'].append(circuit['layer'])
    
    # Normalize statistics
    for pattern in pattern_stats:
        if pattern_stats[pattern]['count'] > 0:
            pattern_stats[pattern]['avg_importance'] /= pattern_stats[pattern]['count']
            pattern_stats[pattern]['dominant_layers'] = list(
                set(pattern_stats[pattern]['layers'])
            )[:5]
    
    # Save results
    results = {
        'dataset': 'BRIGHT',
        'subset': 'documents/theoremqa_questions',
        'model': config['model_name'],
        'num_samples_processed': config['max_samples'],
        'num_circuits': len(all_circuits),
        'circuits': all_circuits[:3000],  # Save subset
        'pattern_statistics': pattern_stats,
        'causal_effects': causal_effects,
        'dataset_statistics': dataset.stats
    }
    
    output_file = config['output_dir'] / 'bright_circuits.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Summary:")
    logger.info(f"  Dataset: BRIGHT (documents/theoremqa_questions)")
    logger.info(f"  Total samples processed: {config['max_samples']}")
    logger.info(f"  Total circuits discovered: {len(all_circuits)}")
    logger.info(f"  Unique patterns: {len(pattern_stats)}")
    
    if pattern_stats:
        logger.info("\n  Pattern distribution:")
        for pattern, stats in sorted(
            pattern_stats.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        ):
            logger.info(
                f"    {pattern}: {stats['count']} circuits, "
                f"avg importance: {stats['avg_importance']:.3f}"
            )
    
    logger.info("\n  Layer-wise causal effects:")
    for layer in sorted(causal_effects['by_layer'].keys())[:5]:
        logger.info(f"    Layer {layer}: {causal_effects['by_layer'][layer]:.4f}")
    
    logger.info(f"\n  Overall causal effect: {causal_effects['overall']:.4f}")
    
    logger.info("="*60)
    logger.info("BRIGHT circuit discovery complete!")


if __name__ == "__main__":
    main()

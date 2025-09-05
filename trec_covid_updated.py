"""
SMA Circuit Discovery for TREC-COVID Dataset - Updated Version
Fixes: Using 'corpus' config name with 'corpus' split, 140K samples
Handles 3 columns (_id, title, text)
Complete implementation with all methods and proper error handling
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
class TRECCOVIDSample:
    """Data structure for TREC-COVID dataset samples"""
    doc_id: str
    title: str
    text: str
    query: str = ""
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = None


class TRECCOVIDDataset(Dataset):
    """
    TREC-COVID dataset loader - Updated for correct corpus configuration
    Handles COVID-19 scientific literature with query-document pairs
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config_name: str = 'corpus',  # Using 'corpus' config as specified
        split: str = 'corpus',  # Using 'corpus' split as specified
        max_samples: int = 140000,  # 140K samples as requested
        cache_dir: str = './cache/trec_covid',
        max_length: int = 256,
        min_doc_length: int = 20,
        generate_queries: bool = True
    ):
        """
        Initialize TREC-COVID dataset with correct configuration
        
        Args:
            tokenizer: HuggingFace tokenizer (preferably biomedical)
            config_name: Config name ('corpus' or 'queries')
            split: Dataset split ('corpus')
            max_samples: Maximum samples to load (default 140K)
            cache_dir: Cache directory
            max_length: Maximum sequence length
            min_doc_length: Minimum document length to consider
            generate_queries: Whether to generate synthetic queries from titles
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_doc_length = min_doc_length
        self.generate_queries = generate_queries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Loading TREC-COVID dataset (config: {config_name}, split: {split})...")
        
        # Load dataset with correct config
        try:
            dataset = load_dataset(
                'BeIR/trec-covid',
                config_name,  # Use 'corpus' config
                split=split,  # Use 'corpus' split
                cache_dir=str(self.cache_dir)
            )
        except Exception as e:
            logger.error(f"Error loading TREC-COVID dataset: {e}")
            logger.info("Attempting alternative loading method...")
            # Alternative loading
            dataset = load_dataset(
                'BeIR/trec-covid',
                name=config_name,
                split=split,
                cache_dir=str(self.cache_dir)
            )
        
        # Process the dataset
        self.data = self._process_dataset(dataset, max_samples)
        logger.info(f"Loaded {len(self.data)} COVID-19 document samples")
        
        # Compute dataset statistics
        self._compute_statistics()
        
        # Initialize COVID-specific patterns
        self._init_covid_patterns()
    
    def _process_dataset(
        self,
        dataset,
        max_samples: int
    ) -> List[TRECCOVIDSample]:
        """
        Process raw dataset into structured format
        Expected columns: _id, title, text
        
        Args:
            dataset: Raw HuggingFace dataset
            max_samples: Maximum number of samples to process
        
        Returns:
            List of processed samples
        """
        processed_data = []
        sample_count = 0
        
        logger.info("Processing TREC-COVID corpus...")
        logger.info(f"Dataset columns: {dataset.column_names if hasattr(dataset, 'column_names') else 'N/A'}")
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing COVID documents")):
            if sample_count >= max_samples:
                break
            
            try:
                # Extract fields according to dataset structure
                # Columns: _id, title, text
                doc_id = str(item.get('_id', f'covid_{idx}'))
                title = item.get('title', '').strip()
                text = item.get('text', '').strip()
                
                # Skip if text is too short
                if not text or len(text.split()) < self.min_doc_length:
                    continue
                
                # Generate query from title or extract key question
                if self.generate_queries and title:
                    # Create query from title
                    query = self._generate_query_from_title(title)
                else:
                    # Extract first sentence as query or use title
                    sentences = text.split('. ')
                    query = sentences[0].strip() if sentences else title
                
                # Ensure both query and document are non-empty
                if not query:
                    query = f"Information about {title[:50]}" if title else "COVID-19 information"
                
                # Create sample
                sample = TRECCOVIDSample(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    query=query,
                    relevance_score=1.0,  # Assume relevance
                    metadata={
                        'source': 'trec_covid',
                        'idx': idx,
                        'has_covid_terms': self._has_covid_terms(title + ' ' + text),
                        'doc_length': len(text.split())
                    }
                )
                
                processed_data.append(sample)
                sample_count += 1
                
            except Exception as e:
                logger.debug(f"Error processing sample {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_data)} samples")
        return processed_data
    
    def _generate_query_from_title(self, title: str) -> str:
        """
        Generate a query from document title
        
        Args:
            title: Document title
        
        Returns:
            Generated query
        """
        # Remove common prefixes/suffixes
        title = title.replace(':', '').replace('?', '')
        
        # Common query patterns for COVID research
        query_patterns = [
            f"What is known about {title}?",
            f"Research on {title}",
            f"Evidence for {title}",
            f"Studies about {title}",
            f"{title} in COVID-19 context"
        ]
        
        # Select pattern based on title characteristics
        if 'treatment' in title.lower() or 'therapy' in title.lower():
            return f"What treatments are discussed regarding {title}?"
        elif 'vaccine' in title.lower():
            return f"What vaccine information is provided about {title}?"
        elif 'transmission' in title.lower() or 'spread' in title.lower():
            return f"How does {title} relate to COVID-19 transmission?"
        elif 'symptom' in title.lower() or 'clinical' in title.lower():
            return f"What clinical features are described in {title}?"
        else:
            return query_patterns[hash(title) % len(query_patterns)]
    
    def _has_covid_terms(self, text: str) -> bool:
        """Check if text contains COVID-19 related terms"""
        covid_terms = [
            'covid', 'coronavirus', 'sars-cov-2', 'pandemic', 'vaccine',
            'spike protein', 'variant', 'transmission', 'quarantine',
            'social distancing', 'ppe', 'ventilator', 'cytokine storm',
            'long covid', 'mrna', 'antibody', 'pcr test', 'antigen'
        ]
        
        text_lower = text.lower()
        return any(term in text_lower for term in covid_terms)
    
    def _init_covid_patterns(self):
        """Initialize COVID-specific research patterns"""
        self.covid_patterns = {
            'clinical': ['symptom', 'diagnosis', 'prognosis', 'mortality', 'severity'],
            'treatment': ['therapy', 'drug', 'medication', 'intervention', 'treatment'],
            'vaccine': ['vaccine', 'immunization', 'antibody', 'immunity', 'efficacy'],
            'transmission': ['transmission', 'spread', 'contagion', 'aerosol', 'droplet'],
            'epidemiology': ['prevalence', 'incidence', 'outbreak', 'cluster', 'wave']
        }
    
    def _compute_statistics(self):
        """Compute dataset statistics with COVID-specific metrics"""
        self.stats = {
            'num_samples': len(self.data),
            'avg_query_length': np.mean([len(s.query.split()) for s in self.data]),
            'avg_doc_length': np.mean([len(s.text.split()) for s in self.data]),
            'avg_title_length': np.mean([len(s.title.split()) for s in self.data if s.title]),
            'max_doc_length': max([len(s.text.split()) for s in self.data]) if self.data else 0,
            'covid_term_ratio': sum([s.metadata.get('has_covid_terms', False) for s in self.data]) / len(self.data) if self.data else 0,
            'docs_with_titles': sum([bool(s.title) for s in self.data])
        }
        
        logger.info("Dataset statistics:")
        for key, value in self.stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with tokenization"""
        sample = self.data[idx]
        
        # Combine title and text for document representation
        document = f"{sample.title}. {sample.text}" if sample.title else sample.text
        
        # Tokenize query and document
        query_encoding = self.tokenizer(
            sample.query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length // 2,
            return_tensors='pt'
        )
        
        doc_encoding = self.tokenizer(
            document,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'doc_id': sample.doc_id,
            'query': sample.query,
            'document': document,
            'title': sample.title,
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
        only_covid: bool = False,
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Get samples for circuit discovery analysis
        
        Args:
            num_samples: Number of samples to return
            only_covid: Whether to filter for COVID-specific content
            random_seed: Random seed for reproducibility
        
        Returns:
            List of samples for analysis
        """
        np.random.seed(random_seed)
        
        # Filter samples if requested
        if only_covid:
            valid_samples = [
                s for s in self.data
                if s.metadata.get('has_covid_terms', False)
            ]
        else:
            valid_samples = self.data
        
        if not valid_samples:
            valid_samples = self.data
        
        indices = np.random.choice(
            len(valid_samples),
            min(num_samples, len(valid_samples)),
            replace=False
        )
        
        samples = []
        for idx in indices:
            sample = valid_samples[idx]
            document = f"{sample.title}. {sample.text}" if sample.title else sample.text
            
            samples.append({
                'query': sample.query,
                'document': document,
                'doc_id': sample.doc_id,
                'title': sample.title,
                'relevance_score': sample.relevance_score,
                'covid_specific': sample.metadata.get('has_covid_terms', False),
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


class TRECCOVIDCircuitDiscovery:
    """
    Circuit discovery module specialized for COVID-19 medical literature patterns
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = 'cuda',
        cache_activations: bool = True
    ):
        """
        Initialize circuit discovery for TREC-COVID
        
        Args:
            model: Pre-trained transformer model (preferably biomedical)
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
        
        # COVID-specific patterns
        self.covid_patterns = {
            'clinical_research': ['symptom', 'diagnosis', 'treatment'],
            'vaccine_development': ['vaccine', 'immunization', 'antibody'],
            'epidemiology': ['transmission', 'spread', 'outbreak'],
            'public_health': ['prevention', 'policy', 'intervention']
        }
        
        logger.info(f"Initialized TREC-COVID circuit discovery with {self.n_layers} layers")
    
    def trace_circuits(
        self,
        query: str,
        document: str,
        covid_features: Optional[Dict[str, Any]] = None,
        threshold: float = 0.02
    ) -> List[Dict[str, Any]]:
        """
        Trace circuits for COVID-19 query-document pair
        
        Args:
            query: Query text
            document: Document text
            covid_features: Additional COVID-specific features
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
        
        # Register hooks
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
                importance = self._compute_medical_importance(
                    query_activations[layer],
                    doc_activations[layer],
                    covid_features
                )
                
                if importance > threshold:
                    # Identify COVID research pattern
                    pattern = self._identify_covid_pattern(query, document)
                    
                    circuit = {
                        'layer': layer,
                        'importance_score': float(importance),
                        'circuit_type': self._classify_circuit_type(layer),
                        'pattern': pattern,
                        'pattern_type': 'medical_covid',
                        'components': ['attention', 'mlp'],
                        'features': covid_features or {}
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
                    if isinstance(output, tuple):
                        self.activation_cache[name] = output[0].detach().cpu()
                    else:
                        self.activation_cache[name] = output.detach().cpu()
            return hook
        
        # Register hooks for each layer
        for i in range(self.n_layers):
            if hasattr(self.model, 'bert'):
                layer = self.model.bert.encoder.layer[i]
            elif hasattr(self.model, 'encoder'):
                layer = self.model.encoder.layer[i]
            else:
                layer = self.model.transformer.h[i]
            
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
    
    def _compute_medical_importance(
        self,
        query_activation: torch.Tensor,
        doc_activation: torch.Tensor,
        covid_features: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute importance score with medical/COVID-specific weighting
        
        Args:
            query_activation: Query activation tensor
            doc_activation: Document activation tensor
            covid_features: Additional COVID features
        
        Returns:
            Importance score
        """
        # Base similarity
        query_norm = F.normalize(query_activation.mean(dim=1), p=2, dim=-1)
        doc_norm = F.normalize(doc_activation.mean(dim=1), p=2, dim=-1)
        similarity = torch.cosine_similarity(query_norm, doc_norm, dim=-1).mean().item()
        
        # Apply medical domain weighting
        if covid_features and covid_features.get('covid_specific', False):
            similarity *= 1.15  # Boost for COVID-specific content
        
        return similarity
    
    def _identify_covid_pattern(self, query: str, document: str) -> str:
        """Identify the type of COVID research pattern"""
        combined_text = (query + ' ' + document).lower()
        
        # Check for different COVID research patterns
        for pattern_type, keywords in self.covid_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                return pattern_type
        
        return 'general_medical'
    
    def _classify_circuit_type(self, layer: int) -> str:
        """Classify circuit type based on layer position"""
        if layer < self.n_layers // 3:
            return 'early_encoding'
        elif layer < 2 * self.n_layers // 3:
            return 'middle_processing'
        else:
            return 'late_integration'
    
    def compute_causal_effects(
        self,
        dataset: TRECCOVIDDataset,
        num_samples: int = 200
    ) -> Dict[str, Any]:
        """
        Compute causal effects through intervention
        
        Args:
            dataset: TREC-COVID dataset
            num_samples: Number of samples to evaluate
        
        Returns:
            Dictionary of causal effects
        """
        logger.info("Computing causal effects for COVID circuits...")
        
        causal_effects = {
            'by_layer': {},
            'by_pattern': {},
            'overall': 0.0
        }
        
        samples = dataset.get_samples_for_analysis(num_samples, only_covid=True)
        
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
                if hasattr(baseline_output, 'last_hidden_state'):
                    baseline_logits = baseline_output.last_hidden_state.mean(dim=1)
                else:
                    baseline_logits = baseline_output[0].mean(dim=1)
            
            # Test interventions
            for layer in range(self.n_layers):
                with self._intervene_on_layer(layer):
                    with torch.no_grad():
                        perturbed_output = self.model(**inputs)
                        if hasattr(perturbed_output, 'last_hidden_state'):
                            perturbed_logits = perturbed_output.last_hidden_state.mean(dim=1)
                        else:
                            perturbed_logits = perturbed_output[0].mean(dim=1)
                    
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
                elif hasattr(self.model, 'encoder'):
                    self.layer_module = self.model.encoder.layer[self.layer]
                else:
                    self.layer_module = self.model.transformer.h[self.layer]
                
                # Store original forward
                self.original_forward = self.layer_module.forward
                
                # Replace with zeroed version
                def zero_forward(*args, **kwargs):
                    output = self.original_forward(*args, **kwargs)
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    else:
                        return torch.zeros_like(output)
                
                self.layer_module.forward = zero_forward
                return self
            
            def __exit__(self, *args):
                # Restore original forward
                self.layer_module.forward = self.original_forward
        
        return InterventionContext(self.model, layer)


def main():
    """Main execution for TREC-COVID circuit discovery"""
    
    # Configuration
    config = {
        'model_name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',  # Biomedical model
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'config_name': 'corpus',  # Use 'corpus' config as specified
        'split': 'corpus',  # Use 'corpus' split
        'max_samples': 140000,  # 140K samples as requested
        'output_dir': Path('./trec_covid_circuits'),
        'threshold': 0.02,  # Lower threshold for medical patterns
        'batch_size': 16,
        'num_workers': 4
    }
    
    logger.info("="*60)
    logger.info("TREC-COVID Circuit Discovery Pipeline")
    logger.info("COVID-19 Medical Literature Analysis")
    logger.info("="*60)
    logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
    
    # Create output directory
    config['output_dir'].mkdir(exist_ok=True, parents=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModel.from_pretrained(config['model_name'])
    
    # Initialize dataset
    logger.info("Initializing TREC-COVID dataset...")
    dataset = TRECCOVIDDataset(
        tokenizer=tokenizer,
        config_name=config['config_name'],
        split=config['split'],
        max_samples=config['max_samples']
    )
    
    # Initialize circuit discovery
    circuit_discovery = TRECCOVIDCircuitDiscovery(
        model=model,
        tokenizer=tokenizer,
        device=config['device']
    )
    
    # Discover circuits
    logger.info("\nDiscovering COVID-specific circuits...")
    all_circuits = []
    
    # Process samples
    analysis_samples = dataset.get_samples_for_analysis(1000, only_covid=True)
    
    for sample in tqdm(analysis_samples, desc="Processing COVID samples"):
        covid_features = {
            'covid_specific': sample.get('covid_specific', False),
            'doc_length': sample['metadata'].get('doc_length', 0)
        }
        
        circuits = circuit_discovery.trace_circuits(
            sample['query'],
            sample['document'],
            covid_features,
            threshold=config['threshold']
        )
        all_circuits.extend(circuits)
    
    logger.info(f"Discovered {len(all_circuits)} circuits")
    
    # Compute causal effects
    logger.info("\nComputing causal effects...")
    causal_effects = circuit_discovery.compute_causal_effects(
        dataset,
        num_samples=200
    )
    
    # Analyze pattern statistics
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
        'dataset': 'TREC-COVID',
        'config': config['config_name'],
        'model': config['model_name'],
        'domain': 'COVID-19 Medical Literature',
        'num_samples_processed': len(dataset.data),
        'num_circuits': len(all_circuits),
        'circuits': all_circuits[:1000],  # Save subset
        'pattern_statistics': pattern_stats,
        'causal_effects': causal_effects,
        'dataset_statistics': dataset.stats
    }
    
    output_file = config['output_dir'] / 'trec_covid_circuits.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print comprehensive summary
    logger.info("\n" + "="*60)
    logger.info("Summary:")
    logger.info(f"  Dataset: TREC-COVID ({config['config_name']}/{config['split']})")
    logger.info(f"  Total samples processed: {len(dataset.data)}")
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
    
    # Top layers by causal effect
    if causal_effects['by_layer']:
        logger.info("\n  Top 5 layers by causal effect:")
        sorted_effects = sorted(
            causal_effects['by_layer'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for layer, effect in sorted_effects:
            logger.info(f"    Layer {layer}: {effect:.4f}")
    
    logger.info(f"\n  Overall causal effect: {causal_effects['overall']:.4f}")
    
    # Dataset characteristics
    logger.info(f"\n  Dataset characteristics:")
    logger.info(f"    COVID term ratio: {dataset.stats.get('covid_term_ratio', 0):.1%}")
    logger.info(f"    Documents with titles: {dataset.stats.get('docs_with_titles', 0)}")
    logger.info(f"    Average document length: {dataset.stats.get('avg_doc_length', 0):.1f} words")
    
    logger.info("="*60)
    logger.info("TREC-COVID circuit discovery complete!")
    logger.info("Medical literature patterns identified successfully")


if __name__ == "__main__":
    main()
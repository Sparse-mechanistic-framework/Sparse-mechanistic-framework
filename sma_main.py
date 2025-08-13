"""
Main Execution Script for SMA Phase 1
Orchestrates the complete mechanistic analysis pipeline
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import logging
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our modules (these should be in the same directory or installed)
from sma_core import (
    IRActivationPatching,
    LogitLensIR,
    ImportanceScorer,
    Circuit,
    verify_implementation
)
from sma_data_loader import (
    load_nfcorpus_for_phase1,
    NFCorpusDataset,
    DataPreparer
)
from sma_visualization import (
    CircuitVisualizer,
    MetricsTracker
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase1Pipeline:
    """
    Complete Phase 1 execution pipeline
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './phase1_results',
        cache_dir: str = './cache'
    ):
        """
        Initialize Phase 1 pipeline
        
        Args:
            model_name: HuggingFace model name
            device: Computation device
            output_dir: Directory for results
            cache_dir: Directory for cached data
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self._initialize_model()
        self._initialize_components()
        
        logger.info(f"Phase 1 Pipeline initialized on {device}")
    
    def _initialize_model(self):
        """Load and prepare the IR model"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Try to load a fine-tuned model, fall back to base model
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=1  # Binary relevance
            )
        except:
            logger.info("Loading base model and adding classification head...")
            base_model = AutoModel.from_pretrained(self.model_name)
            
            # Add classification head for IR
            class IRModel(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.bert = base_model
                    self.config = base_model.config
                    self.dropout = nn.Dropout(0.1)
                    self.classifier = nn.Linear(base_model.config.hidden_size, 1)
                
                def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
                    outputs = self.bert(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    pooled = outputs.pooler_output
                    pooled = self.dropout(pooled)
                    logits = self.classifier(pooled)
                    
                    class Output:
                        def __init__(self, logits):
                            self.logits = logits
                    
                    return Output(logits)
            
            self.model = IRModel(base_model)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _initialize_components(self):
        """Initialize analysis components"""
        logger.info("Initializing analysis components...")
        
        # Activation patching
        self.activation_patcher = IRActivationPatching(
            self.model,
            self.tokenizer,
            self.device
        )
        
        # Logit lens
        self.logit_lens = LogitLensIR(
            self.model,
            self.tokenizer,
            self.device
        )
        
        # Importance scorer
        self.importance_scorer = ImportanceScorer(
            self.model,
            self.activation_patcher,
            self.logit_lens,
            lambda_balance=0.5
        )
        
        # Visualizer
        model_config = {
            'n_layers': self.activation_patcher.n_layers,
            'n_heads': self.activation_patcher.n_heads,
            'hidden_size': self.activation_patcher.hidden_size
        }
        self.visualizer = CircuitVisualizer(
            model_config,
            output_dir=self.output_dir / 'visualizations'
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker(
            output_dir=self.output_dir / 'metrics'
        )
        
        logger.info("Components initialized")
    
    def run_circuit_discovery(
        self,
        data_preparer: DataPreparer,
        num_samples: int = 8000,
        threshold: float = 0.1
    ) -> List[Circuit]:
        """
        Discover circuits in the model
        
        Args:
            data_preparer: Data preparation module
            num_samples: Number of samples to analyze
            threshold: Threshold for circuit detection
        
        Returns:
            List of discovered circuits
        """
        logger.info("=" * 50)
        logger.info("STARTING CIRCUIT DISCOVERY")
        logger.info(f"Threshold: {threshold}, Samples: {num_samples}")
        logger.info("=" * 50)
        
        # Prepare data for circuit discovery
        circuit_data = data_preparer.prepare_circuit_discovery_data(num_samples)
        
        if not circuit_data:
            logger.warning("No suitable data found for circuit discovery!")
            return []
        
        all_circuits = []
        samples_processed = 0
        
        for i, sample in enumerate(tqdm(circuit_data[:num_samples], desc="Discovering circuits")):
            query = sample['query']
            document = sample['document']
            
            try:
                # Trace circuits for this sample
                circuits = self.activation_patcher.trace_circuits(
                    query, document, threshold=threshold
                )
                
                # Log circuits
                for circuit in circuits:
                    self.metrics_tracker.log_circuit(circuit)
                    all_circuits.append(circuit)
                
                samples_processed += 1
                
                # Log progress
                if (i + 1) % 20 == 0:
                    logger.info(f"Processed {i+1} samples, found {len(all_circuits)} circuits")
                    if len(all_circuits) == 0:
                        logger.warning(f"No circuits found yet. Consider lowering threshold (current: {threshold})")
            
            except Exception as e:
                logger.debug(f"Error processing sample {i}: {str(e)}")
                continue
        
        # Deduplicate and rank circuits
        unique_circuits = self._deduplicate_circuits(all_circuits)
        
        logger.info(f"Circuit discovery complete:")
        logger.info(f"  - Samples processed: {samples_processed}")
        logger.info(f"  - Total circuits found: {len(all_circuits)}")
        logger.info(f"  - Unique circuits: {len(unique_circuits)}")
        
        if len(unique_circuits) == 0:
            logger.warning("No circuits discovered! Possible causes:")
            logger.warning("  1. Threshold too high (try lowering from current {threshold})")
            logger.warning("  2. Model not fine-tuned for IR task")
            logger.warning("  3. Insufficient query-document overlap in samples")
        
        return unique_circuits
    
    def _deduplicate_circuits(self, circuits: List[Circuit]) -> List[Circuit]:
        """Remove duplicate circuits and keep highest scoring ones"""
        circuit_dict = {}
        
        for circuit in circuits:
            key = f"{'-'.join(map(str, circuit.layers))}_{'-'.join(circuit.components)}"
            if key not in circuit_dict or circuit.importance_score > circuit_dict[key].importance_score:
                circuit_dict[key] = circuit
        
        return sorted(circuit_dict.values(), key=lambda x: x.importance_score, reverse=True)
    
    def compute_causal_effects(
        self,
        data_preparer: DataPreparer,
        num_samples: int = 1000,
        num_perturbations: int = 5
    ) -> Dict[str, float]:
        """
        Compute causal effects for all components
        
        Args:
            data_preparer: Data preparation module
            num_samples: Number of samples to evaluate
            num_perturbations: Perturbations per sample
        
        Returns:
            Dictionary of causal effects
        """
        logger.info("=" * 50)
        logger.info("COMPUTING CAUSAL EFFECTS")
        logger.info("=" * 50)
        
        # Get evaluation data
        eval_data = data_preparer.dataset.get_samples_for_analysis(
            num_samples, only_positive=True
        )
        
        causal_effects = {}
        
        # Analyze each layer and component
        for layer in tqdm(range(self.activation_patcher.n_layers), desc="Analyzing layers"):
            for component_type in ['attention', 'mlp']:
                component_id = f'layer_{layer}_{component_type}'
                
                effects = []
                for query, doc, _ in eval_data[:30]:  # Subsample for efficiency
                    effect = self.activation_patcher.compute_causal_effect(
                        query, doc, layer, component_type, num_perturbations
                    )
                    effects.append(effect)
                    
                    # Log effect
                    self.metrics_tracker.log_causal_effect(
                        layer, component_type, effect, query
                    )
                
                avg_effect = np.mean(effects)
                causal_effects[component_id] = avg_effect
                
                logger.info(f"{component_id}: causal effect = {avg_effect:.4f}")
        
        return causal_effects
    
    def train_logit_probes(
        self,
        data_preparer: DataPreparer,
        num_samples: int = 2000,
        epochs: int = 7
    ):
        """
        Train logit lens probes
        
        Args:
            data_preparer: Data preparation module
            num_samples: Number of training samples
            epochs: Training epochs
        """
        logger.info("=" * 50)
        logger.info("TRAINING LOGIT LENS PROBES")
        logger.info("=" * 50)
        
        # Get training data
        train_data = data_preparer.dataset.get_samples_for_analysis(num_samples)
        
        # Train probes
        self.logit_lens.train_probes(train_data, epochs=epochs, lr=1e-3)
        
        logger.info("Probe training complete")
    
    def compute_importance_scores(
        self,
        data_preparer: DataPreparer,
        num_samples: int = 4000
    ) -> Dict[str, float]:
        """
        Compute importance scores for all components
        
        Args:
            data_preparer: Data preparation module
            num_samples: Number of evaluation samples
        
        Returns:
            Component importance scores
        """
        logger.info("=" * 50)
        logger.info("COMPUTING IMPORTANCE SCORES")
        logger.info("=" * 50)
        
        # Get evaluation data
        eval_data = data_preparer.dataset.get_samples_for_analysis(num_samples)
        
        # Rank components
        ranked_components = self.importance_scorer.rank_components(
            eval_data, top_k=None
        )
        
        # Log scores
        for component_id, score in ranked_components:
            self.metrics_tracker.log_importance(component_id, score)
        
        # Display top components
        logger.info("\nTop 10 Most Important Components:")
        for i, (component_id, score) in enumerate(ranked_components[:10], 1):
            logger.info(f"{i}. {component_id}: {score:.4f}")
        
        return dict(ranked_components)
    
    def visualize_results(
        self,
        circuits: List[Circuit],
        importance_scores: Dict[str, float],
        sample_data: Optional[Tuple[str, str]] = None
    ):
        """
        Create visualizations of results
        
        Args:
            circuits: Discovered circuits
            importance_scores: Component importance scores
            sample_data: Optional sample for attention visualization
        """
        logger.info("=" * 50)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("=" * 50)
        
        # Circuit graph
        self.visualizer.visualize_circuit_graph(
            circuits[:20],  # Top 20 circuits
            save_name='circuit_graph.html'
        )
        
        # Importance distribution
        self.visualizer.visualize_importance_distribution(
            importance_scores,
            save_name='importance_distribution.png'
        )
        
        # Attention patterns (if sample provided)
        if sample_data:
            query, document = sample_data
            
            # Get attention weights
            with self.activation_patcher.register_hooks():
                inputs = self.tokenizer(
                    query, document,
                    padding=True, truncation=True,
                    max_length=512, return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    _ = self.model(**inputs)
            
            # Visualize attention for middle layer
            middle_layer = self.activation_patcher.n_layers // 2
            attn_key = f'layer_{middle_layer}_attention'
            
            if attn_key in self.activation_patcher.cache.attention_weights:
                query_tokens = query.split()[:20]  # Limit for visualization
                doc_tokens = document.split()[:30]
                
                self.visualizer.visualize_attention_patterns(
                    self.activation_patcher.cache.attention_weights[attn_key],
                    query_tokens, doc_tokens,
                    middle_layer,
                    save_name=f'attention_layer_{middle_layer}.png'
                )
        
        # Summary report
        self.visualizer.create_circuit_summary_report(
            circuits, importance_scores,
            save_name='phase1_summary.html'
        )
        
        # Metrics convergence
        self.metrics_tracker.plot_convergence(save_name='convergence.png')
        
        logger.info("Visualizations complete")
    
    def save_results(
        self,
        circuits: List[Circuit],
        importance_scores: Dict[str, float],
        causal_effects: Dict[str, float]
    ):
        """
        Save all results to disk
        
        Args:
            circuits: Discovered circuits
            importance_scores: Component importance scores
            causal_effects: Causal effect measurements
        """
        logger.info("Saving results...")
        
        # Save circuits
        circuits_data = [c.to_dict() for c in circuits]
        with open(self.output_dir / 'circuits.json', 'w') as f:
            json.dump(circuits_data, f, indent=2)
        
        # Save importance scores
        self.importance_scorer.save_scores(self.output_dir / 'importance_scores.json')
        
        # Save causal effects
        with open(self.output_dir / 'causal_effects.json', 'w') as f:
            json.dump(causal_effects, f, indent=2)
        
        # Save metrics
        self.metrics_tracker.save_metrics('phase1_metrics.json')
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def verify_discoveries(
        self,
        circuits: List[Circuit],
        importance_scores: Dict[str, float],
        min_circuits: int = 3,
        min_importance_variance: float = 0.01
    ) -> bool:
        """
        Verify that discoveries meet minimum criteria
        
        Args:
            circuits: Discovered circuits
            importance_scores: Component importance scores
            min_circuits: Minimum number of circuits required
            min_importance_variance: Minimum variance in importance scores
        
        Returns:
            True if criteria met
        """
        logger.info("=" * 50)
        logger.info("VERIFYING DISCOVERIES")
        logger.info("=" * 50)
        
        success = True
        
        # Check circuit count
        if len(circuits) < min_circuits:
            logger.error(f"❌ Insufficient circuits: {len(circuits)} < {min_circuits}")
            success = False
        else:
            logger.info(f"✓ Found {len(circuits)} circuits (>= {min_circuits})")
        
        # Check importance score variance
        scores = list(importance_scores.values())
        if scores:
            variance = np.var(scores)
            if variance < min_importance_variance:
                logger.error(f"❌ Low importance variance: {variance:.6f} < {min_importance_variance}")
                success = False
            else:
                logger.info(f"✓ Importance variance: {variance:.6f} (>= {min_importance_variance})")
        
        # Check for interpretable circuits
        interpretable_circuits = [c for c in circuits if c.importance_score > 0.1]
        if len(interpretable_circuits) == 0:
            logger.error("❌ No interpretable circuits found (importance > 0.1)")
            success = False
        else:
            logger.info(f"✓ Found {len(interpretable_circuits)} interpretable circuits")
        
        if success:
            logger.info("✅ ALL VERIFICATION CHECKS PASSED")
        else:
            logger.warning("⚠️ Some verification checks failed - review results")
        
        return success


def main():
    """
    Main execution function for Phase 1
    """
    parser = argparse.ArgumentParser(description='SMA Phase 1: Core Implementation with NFCorpus')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       help='Model name from HuggingFace')
    parser.add_argument('--num-samples', type=int, default=40000,
                       help='Number of NFCorpus samples to load')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Computation device')
    parser.add_argument('--output-dir', type=str, default='./phase1_results',
                       help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                       help='Cache directory')
    parser.add_argument('--skip-verification', action='store_true',
                       help='Skip initial verification tests')
    
    args = parser.parse_args()
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("SMA PHASE 1: CORE IMPLEMENTATION")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Samples: {args.num_samples}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = Phase1Pipeline(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    # Load data
    logger.info("\n📊 Loading NFCorpus dataset...")
    dataset, data_preparer = load_nfcorpus_for_phase1(
        pipeline.tokenizer,
        max_samples=args.num_samples,
        cache_dir=args.cache_dir,
        split='test'  # NFCorpus uses test split for evaluation
    )
    
    # Run verification if not skipped
    if not args.skip_verification:
        logger.info("\n🔍 Running implementation verification...")
        sample_data = dataset.get_samples_for_analysis(5)
        if not verify_implementation(pipeline.model, pipeline.tokenizer, sample_data):
            logger.error("Verification failed! Fix issues before proceeding.")
            return
    
    # Start analysis
    start_time = time.time()
    
    # 1. Circuit Discovery
    logger.info("\n🔬 Phase 1.1: Circuit Discovery")
    circuits = pipeline.run_circuit_discovery(
        data_preparer,
        num_samples=min(3000, args.num_samples),  # Increased samples for NFCorpus
        threshold=0.04  # Lower threshold for medical domain patterns
    )
    
    # 2. Causal Effects
    logger.info("\n⚡ Phase 1.2: Computing Causal Effects")
    causal_effects = pipeline.compute_causal_effects(
        data_preparer,
        num_samples=min(1000, args.num_samples // 2),
        num_perturbations=5
    )
    
    # 3. Train Logit Probes
    logger.info("\n🎯 Phase 1.3: Training Logit Lens Probes")
    pipeline.train_logit_probes(
        data_preparer,
        num_samples=min(2000, args.num_samples),
        epochs=8
    )
    
    # 4. Importance Scoring
    logger.info("\n📈 Phase 1.4: Computing Importance Scores")
    importance_scores = pipeline.compute_importance_scores(
        data_preparer,
        num_samples=min(4000, args.num_samples // 2)
    )
    
    # 5. Visualizations
    logger.info("\n🎨 Phase 1.5: Creating Visualizations")
    sample_query, sample_doc, _ = dataset.get_samples_for_analysis(1)[0]
    pipeline.visualize_results(
        circuits,
        importance_scores,
        sample_data=(sample_query, sample_doc)
    )
    
    # 6. Save Results
    logger.info("\n💾 Phase 1.6: Saving Results")
    pipeline.save_results(circuits, importance_scores, causal_effects)
    
    # 7. Verification
    logger.info("\n✅ Phase 1.7: Final Verification")
    success = pipeline.verify_discoveries(
        circuits,
        importance_scores,
        min_circuits=4,
        min_importance_variance=0.001
    )
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Circuits discovered: {len(circuits)}")
    logger.info(f"Components analyzed: {len(importance_scores)}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Verification: {'PASSED ✅' if success else 'FAILED ❌'}")
    logger.info("=" * 60)
    
    if success:
        logger.info("\n🎉 Phase 1 completed successfully!")
        logger.info("You can now proceed to Phase 2: Pruning Implementation")
    else:
        logger.warning("\n⚠️ Phase 1 completed with warnings. Review results before proceeding.")


if __name__ == "__main__":
    main()
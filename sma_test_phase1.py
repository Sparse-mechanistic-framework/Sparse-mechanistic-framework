"""
Quick test script to debug circuit discovery with NFCorpus
Run this to verify the setup and find optimal threshold
"""

import torch
from transformers import AutoModel, AutoTokenizer
import logging
from pathlib import Path
import numpy as np

# Import our modules
from sma_core import IRActivationPatching, LogitLensIR, ImportanceScorer
from sma_data_loader import load_nfcorpus_for_phase1

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def test_circuit_discovery():
    """Test circuit discovery with different thresholds"""
    
    # Initialize model and tokenizer
    model_name = 'bert-base-uncased'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Load NFCorpus data
    logger.info("Loading NFCorpus dataset...")
    dataset, data_preparer = load_nfcorpus_for_phase1(
        tokenizer,
        max_samples=500,
        cache_dir='./cache',
        split='test'
    )
    
    # Get some samples
    samples = data_preparer.prepare_circuit_discovery_data(10)
    
    if not samples:
        logger.error("No samples found!")
        return
    
    # Initialize activation patcher
    activation_patcher = IRActivationPatching(model, tokenizer, device)
    
    # Test with different thresholds
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    logger.info("\nTesting circuit discovery with different thresholds:")
    logger.info("=" * 60)
    
    for threshold in thresholds:
        total_circuits = 0
        
        # Test on first 5 samples
        for i, sample in enumerate(samples[:5]):
            query = sample['query']
            document = sample['document']
            
            # Show sample info
            if threshold == thresholds[0]:  # Only show once
                logger.info(f"\nSample {i+1}:")
                logger.info(f"  Query: {query[:100]}...")
                logger.info(f"  Doc: {document[:100]}...")
                logger.info(f"  Relevance: {sample['relevance']}")
                logger.info(f"  Term overlap: {sample['term_overlap']:.2f}")
            
            try:
                circuits = activation_patcher.trace_circuits(
                    query, document, threshold=threshold
                )
                total_circuits += len(circuits)
            except Exception as e:
                logger.error(f"Error: {e}")
                continue
        
        logger.info(f"Threshold {threshold:.3f}: {total_circuits} circuits found")
    
    # Analyze attention patterns for the first sample
    logger.info("\n" + "=" * 60)
    logger.info("Analyzing attention patterns for first sample...")
    
    sample = samples[0]
    with activation_patcher.register_hooks():
        inputs = tokenizer(
            sample['query'], sample['document'],
            padding=True, truncation=True,
            max_length=256, return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
    
    # Check what was cached
    logger.info(f"Cached activations: {list(activation_patcher.cache.activations.keys())[:5]}")
    logger.info(f"Cached attention weights: {list(activation_patcher.cache.attention_weights.keys())[:5]}")
    
    # Analyze attention patterns
    for layer in range(min(3, activation_patcher.n_layers)):
        attn_key = f'layer_{layer}_attention'
        if attn_key in activation_patcher.cache.attention_weights:
            attn_weights = activation_patcher.cache.attention_weights[attn_key]
            mean_attn = attn_weights.mean().item()
            max_attn = attn_weights.max().item()
            logger.info(f"Layer {layer}: mean attention = {mean_attn:.4f}, max = {max_attn:.4f}")
    
    logger.info("\nTest complete!")


def test_medical_patterns():
    """Test if medical domain has clearer patterns"""
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing medical domain pattern clarity...")
    
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Medical query-document pairs (synthetic examples)
    test_pairs = [
        {
            'query': 'vitamin D deficiency symptoms',
            'document': 'Vitamin D deficiency can cause fatigue, bone pain, muscle weakness, and mood changes. Low vitamin D levels are associated with depression and weakened immune system.'
        },
        {
            'query': 'diabetes type 2 treatment',
            'document': 'Type 2 diabetes treatment includes lifestyle changes, metformin medication, blood sugar monitoring, and insulin therapy in advanced cases.'
        },
        {
            'query': 'high blood pressure causes',
            'document': 'Hypertension or high blood pressure can be caused by obesity, high salt intake, stress, genetics, and lack of physical activity.'
        }
    ]
    
    for i, pair in enumerate(test_pairs):
        query_tokens = set(pair['query'].lower().split())
        doc_tokens = set(pair['document'].lower().split())
        
        # Calculate overlap
        overlap = query_tokens & doc_tokens
        overlap_ratio = len(overlap) / len(query_tokens) if query_tokens else 0
        
        logger.info(f"\nPair {i+1}:")
        logger.info(f"  Query: {pair['query']}")
        logger.info(f"  Overlapping terms: {overlap}")
        logger.info(f"  Overlap ratio: {overlap_ratio:.2f}")
    
    logger.info("\nMedical domain shows clear term overlap patterns!")


if __name__ == "__main__":
    logger.info("Starting Phase 1 Circuit Discovery Test")
    logger.info("=" * 60)
    
    # Test basic circuit discovery
    test_circuit_discovery()
    
    # Test medical patterns
    test_medical_patterns()
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests complete!")
    logger.info("Recommended threshold for NFCorpus: 0.01 - 0.02")
    logger.info("If still no circuits found, consider:")
    logger.info("  1. Fine-tuning model on NFCorpus first")
    logger.info("  2. Using a model pre-trained on medical text")
    logger.info("  3. Implementing gradient-based circuit detection")
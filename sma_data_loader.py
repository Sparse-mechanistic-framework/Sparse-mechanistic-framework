"""
NFCorpus Data Loader and Preprocessing for SMA
Handles data loading, preprocessing, and batch generation for NFCorpus medical IR dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Any
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
import logging
from tqdm.auto import tqdm
from pathlib import Path
import json
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)


class NFCorpusDataset(Dataset):
    """
    NFCorpus dataset for IR experiments
    Medical/nutrition domain with graded relevance
    """
    
    def __init__(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        min_doc_length: int = 10,
        max_doc_length: int = 512,
        relevance_threshold: float = 1.0
    ):
        """
        Initialize NFCorpus dataset
        
        Args:
            split: Dataset split ('train', 'dev', 'test')
            max_samples: Maximum number of samples to load
            cache_dir: Directory for caching processed data
            min_doc_length: Minimum document length in tokens
            max_doc_length: Maximum document length in tokens
            relevance_threshold: Minimum relevance score to consider positive
        """
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.relevance_threshold = relevance_threshold
        
        # Load data
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples from NFCorpus {split}")
        logger.info(f"Corpus size: {len(self.corpus)}, Queries: {len(self.queries)}")
    
    def _load_data(self) -> List[Dict]:
        """Load and preprocess NFCorpus data"""
        cache_file = self.cache_dir / f'nfcorpus_{self.split}_{self.max_samples}.pkl'
        
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.corpus = cached_data['corpus']
                self.queries = cached_data['queries']
                self.qrels = cached_data['qrels']
                return cached_data['data']
        
        logger.info("Loading NFCorpus from HuggingFace datasets...")
        
        # Load corpus
        corpus_data = load_dataset("mteb/nfcorpus", "corpus", split="corpus")
        for item in corpus_data:
            doc_id = item['_id']
            # Combine title and text for document content
            doc_text = f"{item.get('title', '')} {item.get('text', '')}".strip()
            self.corpus[doc_id] = doc_text
        
        # Load queries
        queries_data = load_dataset("mteb/nfcorpus", "queries", split="queries")
        for item in queries_data:
            query_id = item['_id']
            self.queries[query_id] = item['text']
        
        # Load relevance judgments (qrels)
        qrels_data = load_dataset("mteb/nfcorpus", "default", split=self.split)
        for item in qrels_data:
            query_id = item['query-id']
            corpus_id = item['corpus-id']
            score = item['score']
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {}
            self.qrels[query_id][corpus_id] = score
        
        # Create query-document pairs with relevance scores
        processed_data = []
        
        for query_id, doc_scores in tqdm(self.qrels.items(), desc="Processing qrels"):
            if query_id not in self.queries:
                continue
                
            query_text = self.queries[query_id]
            
            # Add positive samples (relevant documents)
            for doc_id, score in doc_scores.items():
                if doc_id not in self.corpus:
                    continue
                    
                doc_text = self.corpus[doc_id]
                
                # Check document length
                doc_tokens = doc_text.split()
                if len(doc_tokens) < self.min_doc_length:
                    continue
                if len(doc_tokens) > self.max_doc_length:
                    doc_text = ' '.join(doc_tokens[:self.max_doc_length])
                
                # Normalize relevance score (NFCorpus uses 0-2 scale)
                normalized_score = score / 2.0
                
                processed_data.append({
                    'query_id': query_id,
                    'query': query_text,
                    'doc_id': doc_id,
                    'document': doc_text,
                    'relevance': normalized_score,
                    'original_score': score
                })
            
            # Add some negative samples (non-relevant documents)
            # Sample random documents that aren't in the qrels for this query
            all_doc_ids = set(self.corpus.keys())
            relevant_doc_ids = set(doc_scores.keys())
            non_relevant_ids = list(all_doc_ids - relevant_doc_ids)
            
            # Sample up to 5 negative documents per query
            num_negatives = min(7, len(non_relevant_ids))
            if num_negatives > 0:
                negative_samples = np.random.choice(non_relevant_ids, num_negatives, replace=False)
                for doc_id in negative_samples:
                    doc_text = self.corpus[doc_id]
                    doc_tokens = doc_text.split()
                    
                    if len(doc_tokens) < self.min_doc_length:
                        continue
                    if len(doc_tokens) > self.max_doc_length:
                        doc_text = ' '.join(doc_tokens[:self.max_doc_length])
                    
                    processed_data.append({
                        'query_id': query_id,
                        'query': query_text,
                        'doc_id': doc_id,
                        'document': doc_text,
                        'relevance': 0.0,
                        'original_score': 0
                    })
            
            if self.max_samples and len(processed_data) >= self.max_samples:
                processed_data = processed_data[:self.max_samples]
                break
        
        # Shuffle data
        np.random.shuffle(processed_data)
        
        # Cache processed data
        cache_data = {
            'corpus': self.corpus,
            'queries': self.queries,
            'qrels': self.qrels,
            'data': processed_data
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Created {len(processed_data)} query-document pairs")
        positive_samples = sum(1 for d in processed_data if d['relevance'] > 0)
        logger.info(f"Positive samples: {positive_samples}, Negative samples: {len(processed_data) - positive_samples}")
        
        return processed_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
    
    def get_samples_for_analysis(
        self,
        num_samples: int = 3000,
        only_positive: bool = False
    ) -> List[Tuple[str, str, float]]:
        """
        Get samples formatted for mechanistic analysis
        
        Args:
            num_samples: Number of samples to return
            only_positive: Whether to return only positive samples
        
        Returns:
            List of (query, document, relevance) tuples
        """
        if only_positive:
            samples = [d for d in self.data if d['relevance'] > 0]
        else:
            samples = self.data
        
        if len(samples) > num_samples:
            samples = np.random.choice(samples, num_samples, replace=False)
        
        return [
            (s['query'], s['document'], s['relevance'])
            for s in samples
        ]
    
    def get_hard_negatives(
        self,
        num_samples: int = 300
    ) -> List[Tuple[str, str, str]]:
        """
        Get hard negative samples (same query, different documents)
        
        Returns:
            List of (query, positive_doc, negative_doc) tuples
        """
        # Group by query
        query_groups = defaultdict(list)
        for sample in self.data:
            query_groups[sample['query']].append(sample)
        
        hard_negatives = []
        
        for query, docs in query_groups.items():
            positive = [d for d in docs if d['relevance'] > 0]
            negative = [d for d in docs if d['relevance'] == 0]
            
            if positive and negative:
                pos_doc = positive[0]['document']
                neg_doc = negative[0]['document']
                hard_negatives.append((query, pos_doc, neg_doc))
            
            if len(hard_negatives) >= num_samples:
                break
        
        return hard_negatives


class IRDataCollator:
    """
    Data collator for IR model training/evaluation
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        padding: str = 'max_length'
    ):
        """
        Initialize data collator
        
        Args:
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            padding: Padding strategy
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
    
    def __call__(
        self,
        batch: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            batch: List of sample dictionaries
        
        Returns:
            Collated batch ready for model input
        """
        queries = [sample['query'] for sample in batch]
        documents = [sample['document'] for sample in batch]
        relevances = [sample['relevance'] for sample in batch]
        
        # Tokenize query-document pairs
        encoded = self.tokenizer(
            queries,
            documents,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Add relevance labels
        encoded['labels'] = torch.tensor(relevances, dtype=torch.float)
        
        return encoded


class DataPreparer:
    """
    Prepare data for different phases of analysis
    """
    
    def __init__(
        self,
        dataset: NFCorpusDataset,
        tokenizer: Any
    ):
        """
        Initialize data preparer
        
        Args:
            dataset: NFCorpus dataset
            tokenizer: Model tokenizer
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def prepare_circuit_discovery_data(
        self,
        num_samples: int = 2000
    ) -> List[Dict]:
        """
        Prepare data for circuit discovery
        Focus on clear positive examples
        
        Args:
            num_samples: Number of samples
        
        Returns:
            List of prepared samples
        """
        # Get positive samples with high confidence
        positive_samples = [
            s for s in self.dataset.data 
            if s['relevance'] > 0
        ]
        
        if len(positive_samples) > num_samples:
            positive_samples = np.random.choice(
                positive_samples, num_samples, replace=False
            )
        
        prepared = []
        for sample in positive_samples:
            # Tokenize to check for query term overlap
            query_tokens = set(sample['query'].lower().split())
            doc_tokens = set(sample['document'].lower().split())
            
            # Compute overlap score
            overlap = len(query_tokens & doc_tokens) / len(query_tokens)
            
            prepared.append({
                'query': sample['query'],
                'document': sample['document'],
                'relevance': sample['relevance'],
                'term_overlap': overlap,
                'query_length': len(query_tokens),
                'doc_length': len(doc_tokens)
            })
        
        # Sort by term overlap (higher overlap = clearer signal)
        prepared.sort(key=lambda x: x['term_overlap'], reverse=True)
        
        return prepared
    
    def prepare_importance_scoring_data(
        self,
        num_samples: int = 1000
    ) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
        """
        Prepare balanced data for importance scoring
        
        Args:
            num_samples: Total number of samples
        
        Returns:
            Training and validation data
        """
        # Get balanced samples
        samples = self.dataset.get_samples_for_analysis(num_samples)
        
        # Split into train/val
        split_idx = int(len(samples) * 0.8)
        train_data = samples[:split_idx]
        val_data = samples[split_idx:]
        
        return train_data, val_data
    
    def prepare_counterfactual_pairs(
        self,
        num_pairs: int = 1000
    ) -> List[Dict]:
        """
        Prepare counterfactual query-document pairs
        
        Args:
            num_pairs: Number of pairs to prepare
        
        Returns:
            List of counterfactual pairs
        """
        hard_negatives = self.dataset.get_hard_negatives(num_pairs)
        
        counterfactuals = []
        for query, pos_doc, neg_doc in hard_negatives:
            counterfactuals.append({
                'query': query,
                'original_doc': pos_doc,
                'counterfactual_doc': neg_doc,
                'original_relevance': 1.0,
                'counterfactual_relevance': 0.0
            })
        
        return counterfactuals
    
    def create_dataloaders(
        self,
        batch_size: int = 16,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for training and validation
        
        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
        
        Returns:
            Train and validation DataLoaders
        """
        # Split dataset
        train_size = int(len(self.dataset) * 0.8)
        val_size = len(self.dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create collator
        collator = IRDataCollator(self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator
        )
        
        return train_loader, val_loader


def load_nfcorpus_for_phase1(
    tokenizer: Any,
    max_samples: int = 50000,
    cache_dir: Optional[str] = None,
    split: str = 'test'
) -> Tuple[NFCorpusDataset, DataPreparer]:
    """
    Convenience function to load NFCorpus for Phase 1
    
    Args:
        tokenizer: Model tokenizer
        max_samples: Maximum samples to load
        cache_dir: Cache directory
        split: Dataset split to use
    
    Returns:
        Dataset and data preparer
    """
    logger.info("Loading NFCorpus dataset for Phase 1...")
    
    # Load dataset
    dataset = NFCorpusDataset(
        split=split,
        max_samples=max_samples,
        cache_dir=cache_dir
    )
    
    # Create data preparer
    preparer = DataPreparer(dataset, tokenizer)
    
    logger.info(f"Dataset ready with {len(dataset)} samples")
    logger.info(f"Unique queries: {len(dataset.queries)}")
    logger.info(f"Unique documents: {len(dataset.corpus)}")
    
    return dataset, preparer
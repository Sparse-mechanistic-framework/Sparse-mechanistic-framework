"""
cross_domain_evaluation.py
Evaluate pruned models across different IR domains (medical, legal, general)
Senior AI Developer Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import traceback
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('cross_domain_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DomainDataset(Dataset):
    """Generic dataset for different domains"""
    
    def __init__(
        self,
        domain: str,
        tokenizer: Any,
        max_samples: int = 1000,
        max_length: int = 256,
        cache_dir: str = './cache'
    ):
        """
        Initialize domain-specific dataset
        
        Args:
            domain: Domain name ('medical', 'legal', 'general')
            tokenizer: Tokenizer for encoding
            max_samples: Maximum samples to load
            max_length: Maximum sequence length
            cache_dir: Cache directory
        """
        self.domain = domain
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.data = self._load_domain_data()
        logger.info(f"Loaded {len(self.data)} samples for {domain} domain")
    
    def _load_domain_data(self) -> List[Dict]:
        """Load domain-specific data"""
        try:
            if self.domain == 'medical':
                return self._load_medical_data()
            elif self.domain == 'legal':
                return self._load_legal_data()
            elif self.domain == 'general':
                return self._load_general_data()
            else:
                raise ValueError(f"Unknown domain: {self.domain}")
                
        except Exception as e:
            logger.error(f"Failed to load {self.domain} data: {str(e)}")
            logger.warning(f"Using synthetic data for {self.domain}")
            return self._generate_synthetic_data()
    
    def _load_medical_data(self) -> List[Dict]:
        """Load medical domain data (NFCorpus)"""
        try:
            # Load NFCorpus for medical domain
            corpus_data = load_dataset("mteb/nfcorpus", "corpus", split="corpus")
            queries_data = load_dataset("mteb/nfcorpus", "queries", split="queries")
            qrels_data = load_dataset("mteb/nfcorpus", "default", split="test")
            
            corpus = {}
            for item in corpus_data:
                doc_id = item['_id']
                corpus[doc_id] = f"{item.get('title', '')} {item.get('text', '')}"
            
            queries = {}
            for item in queries_data:
                queries[item['_id']] = item['text']
            
            data = []
            for item in qrels_data:
                if len(data) >= self.max_samples:
                    break
                    
                query_id = item['query-id']
                doc_id = item['corpus-id']
                score = item['score'] / 2.0  # Normalize to [0, 1]
                
                if query_id in queries and doc_id in corpus:
                    data.append({
                        'query': queries[query_id],
                        'document': corpus[doc_id][:1000],  # Truncate long docs
                        'relevance': score
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load NFCorpus: {str(e)}")
            return self._generate_synthetic_medical_data()
    
    def _load_legal_data(self) -> List[Dict]:
        """Load legal domain data"""
        try:
            # Try to load CaseHOLD or similar legal dataset
            # For now, generate synthetic legal data
            return self._generate_synthetic_legal_data()
            
        except Exception as e:
            logger.error(f"Failed to load legal data: {str(e)}")
            return self._generate_synthetic_legal_data()
    
    def _load_general_data(self) -> List[Dict]:
        """Load general domain data (MS MARCO subset)"""
        try:
            # For demonstration, use synthetic general data
            # In production, load MS MARCO or similar
            return self._generate_synthetic_general_data()
            
        except Exception as e:
            logger.error(f"Failed to load general data: {str(e)}")
            return self._generate_synthetic_general_data()
    
    def _generate_synthetic_medical_data(self) -> List[Dict]:
        """Generate synthetic medical query-document pairs"""
        medical_queries = [
            "diabetes type 2 treatment options",
            "high blood pressure symptoms",
            "COVID-19 vaccine side effects",
            "vitamin D deficiency diagnosis",
            "chronic pain management strategies",
            "heart disease prevention methods",
            "cancer screening guidelines",
            "mental health therapy options",
            "arthritis pain relief",
            "sleep disorder treatments"
        ]
        
        medical_docs = [
            "Type 2 diabetes can be managed through lifestyle changes including diet modification, regular exercise, and weight loss. Medications like metformin are commonly prescribed.",
            "High blood pressure symptoms include headaches, shortness of breath, and nosebleeds. However, many people have no symptoms.",
            "Common COVID-19 vaccine side effects include pain at injection site, fatigue, headache, and mild fever lasting 1-2 days.",
            "Vitamin D deficiency is diagnosed through blood tests measuring 25-hydroxyvitamin D levels. Levels below 20 ng/mL indicate deficiency.",
            "Chronic pain management includes physical therapy, medications, cognitive behavioral therapy, and relaxation techniques.",
            "Heart disease prevention involves maintaining healthy weight, regular exercise, balanced diet, not smoking, and managing stress.",
            "Cancer screening guidelines vary by age and risk factors. Common screenings include mammograms, colonoscopy, and skin checks.",
            "Mental health therapy options include cognitive behavioral therapy, psychodynamic therapy, and medication management.",
            "Arthritis pain can be managed with anti-inflammatory medications, physical therapy, hot/cold therapy, and joint protection.",
            "Sleep disorders are treated based on type. Options include sleep hygiene, CPAP for apnea, and cognitive behavioral therapy for insomnia."
        ]
        
        data = []
        for i in range(min(self.max_samples, len(medical_queries) * 10)):
            q_idx = i % len(medical_queries)
            d_idx = np.random.randint(len(medical_docs))
            
            # Relevance based on query-document match
            relevance = 1.0 if q_idx == d_idx else np.random.uniform(0, 0.5)
            
            data.append({
                'query': medical_queries[q_idx],
                'document': medical_docs[d_idx],
                'relevance': relevance
            })
        
        return data
    
    def _generate_synthetic_legal_data(self) -> List[Dict]:
        """Generate synthetic legal query-document pairs"""
        legal_queries = [
            "contract breach remedies",
            "intellectual property infringement",
            "employment discrimination laws",
            "personal injury statute limitations",
            "bankruptcy chapter 7 requirements",
            "criminal defense strategies",
            "family law custody factors",
            "real estate closing process",
            "tax law deductions",
            "immigration visa types"
        ]
        
        legal_docs = [
            "Contract breach remedies include compensatory damages, specific performance, and rescission. Courts determine appropriate remedy based on breach severity.",
            "Intellectual property infringement involves unauthorized use of patents, trademarks, or copyrights. Remedies include injunctions and damages.",
            "Employment discrimination based on protected characteristics is prohibited under federal law including Title VII and ADA.",
            "Personal injury claims must be filed within statute of limitations, typically 2-3 years from injury date depending on jurisdiction.",
            "Chapter 7 bankruptcy requires means test, credit counseling, and liquidation of non-exempt assets to discharge debts.",
            "Criminal defense strategies include challenging evidence admissibility, negotiating plea bargains, and presenting affirmative defenses.",
            "Child custody decisions consider best interests including stability, parental fitness, and child preferences in some cases.",
            "Real estate closing involves title search, inspections, financing approval, and deed transfer at settlement.",
            "Tax deductions reduce taxable income. Common deductions include mortgage interest, charitable contributions, and business expenses.",
            "US immigration visas include family-based, employment-based, and diversity visas with varying requirements and quotas."
        ]
        
        data = []
        for i in range(min(self.max_samples, len(legal_queries) * 10)):
            q_idx = i % len(legal_queries)
            d_idx = np.random.randint(len(legal_docs))
            
            relevance = 1.0 if q_idx == d_idx else np.random.uniform(0, 0.5)
            
            data.append({
                'query': legal_queries[q_idx],
                'document': legal_docs[d_idx],
                'relevance': relevance
            })
        
        return data
    
    def _generate_synthetic_general_data(self) -> List[Dict]:
        """Generate synthetic general domain query-document pairs"""
        general_queries = [
            "how to bake chocolate cake",
            "best tourist attractions Paris",
            "smartphone battery life tips",
            "climate change effects",
            "stock market investing basics",
            "home workout routines",
            "python programming tutorials",
            "sustainable living tips",
            "car maintenance schedule",
            "online learning platforms"
        ]
        
        general_docs = [
            "Chocolate cake requires flour, sugar, cocoa powder, eggs, and butter. Bake at 350°F for 30-35 minutes until toothpick comes out clean.",
            "Paris attractions include Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Arc de Triomphe, and Champs-Élysées.",
            "Extend smartphone battery by reducing brightness, closing unused apps, disabling location services, and using power saving mode.",
            "Climate change causes rising temperatures, melting ice caps, extreme weather events, and ecosystem disruptions globally.",
            "Stock market investing basics: diversify portfolio, understand risk tolerance, research companies, and consider index funds for beginners.",
            "Home workouts include bodyweight exercises like push-ups, squats, lunges, planks, and burpees for full-body fitness.",
            "Python programming fundamentals cover variables, data types, functions, loops, and object-oriented programming concepts.",
            "Sustainable living involves reducing waste, conserving energy, using renewable resources, and supporting eco-friendly products.",
            "Regular car maintenance includes oil changes every 5000 miles, tire rotations, brake inspections, and fluid checks.",
            "Popular online learning platforms include Coursera, edX, Udemy, Khan Academy, and LinkedIn Learning for various subjects."
        ]
        
        data = []
        for i in range(min(self.max_samples, len(general_queries) * 10)):
            q_idx = i % len(general_queries)
            d_idx = np.random.randint(len(general_docs))
            
            relevance = 1.0 if q_idx == d_idx else np.random.uniform(0, 0.5)
            
            data.append({
                'query': general_queries[q_idx],
                'document': general_docs[d_idx],
                'relevance': relevance
            })
        
        return data
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generic synthetic data generation"""
        if self.domain == 'medical':
            return self._generate_synthetic_medical_data()
        elif self.domain == 'legal':
            return self._generate_synthetic_legal_data()
        else:
            return self._generate_synthetic_general_data()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        
        encoded = self.tokenizer(
            sample['query'],
            sample['document'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['relevance'], dtype=torch.float)
        }


class CrossDomainEvaluator:
    """Evaluate models across different domains"""
    
    def __init__(
        self,
        models_dir: Path,
        output_dir: Path,
        device: str = 'cuda'
    ):
        """
        Initialize cross-domain evaluator
        
        Args:
            models_dir: Directory containing pruned models
            output_dir: Output directory for results
            device: Device for computation
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        
        self.results = defaultdict(lambda: defaultdict(dict))
        self.domain_importance = {}
    
    def evaluate_model_on_domain(
        self,
        model: nn.Module,
        domain_loader: DataLoader,
        domain_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model on specific domain
        
        Args:
            model: Model to evaluate
            domain_loader: Domain-specific data loader
            domain_name: Name of domain
            
        Returns:
            Evaluation metrics
        """
        try:
            model.eval()
            model.to(self.device)
            
            predictions = []
            labels = []
            losses = []
            inference_times = []
            
            with torch.no_grad():
                for batch in tqdm(domain_loader, desc=f"Evaluating {domain_name}", leave=False):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # Measure inference time
                    start_time = time.perf_counter()
                    
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    inference_time = time.perf_counter() - start_time
                    inference_times.append(inference_time)
                    
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
            
            # Correlation
            if len(predictions) > 1:
                correlation = np.corrcoef(predictions, labels)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
            else:
                correlation = 0
            
            # MSE
            mse = np.mean((predictions - labels) ** 2)
            
            # RMSE
            rmse = np.sqrt(mse)
            
            # MAE
            mae = np.mean(np.abs(predictions - labels))
            
            return {
                'loss': np.mean(losses),
                'correlation': correlation,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'avg_inference_time': np.mean(inference_times),
                'total_samples': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed for {domain_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'loss': float('inf'),
                'correlation': 0,
                'mse': float('inf'),
                'rmse': float('inf'),
                'mae': float('inf'),
                'avg_inference_time': 0,
                'total_samples': 0
            }
    
    def analyze_domain_shift(
        self,
        base_results: Dict[str, float],
        domain_results: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Analyze performance shift across domains
        
        Args:
            base_results: Results on original domain
            domain_results: Results on target domain
            
        Returns:
            Domain shift metrics
        """
        shift_metrics = {}
        
        # Performance drop
        shift_metrics['correlation_drop'] = base_results['correlation'] - domain_results['correlation']
        shift_metrics['mse_increase'] = domain_results['mse'] - base_results['mse']
        
        # Relative changes
        if base_results['correlation'] > 0:
            shift_metrics['correlation_retention'] = domain_results['correlation'] / base_results['correlation']
        else:
            shift_metrics['correlation_retention'] = 0
        
        if base_results['mse'] > 0:
            shift_metrics['mse_ratio'] = domain_results['mse'] / base_results['mse']
        else:
            shift_metrics['mse_ratio'] = float('inf')
        
        # Domain robustness score (0-1, higher is better)
        shift_metrics['robustness_score'] = max(0, min(1, shift_metrics['correlation_retention']))
        
        return shift_metrics
    
    def run_cross_domain_evaluation(
        self,
        tokenizer: Any,
        domains: List[str] = ['medical', 'legal', 'general'],
        max_samples_per_domain: int = 1000,
        batch_size: int = 16
    ):
        """
        Run evaluation across all domains
        
        Args:
            tokenizer: Tokenizer for encoding
            domains: List of domains to evaluate
            max_samples_per_domain: Maximum samples per domain
            batch_size: Batch size for evaluation
        """
        logger.info("="*60)
        logger.info("CROSS-DOMAIN EVALUATION")
        logger.info("="*60)
        
        # Load domain datasets
        domain_loaders = {}
        for domain in domains:
            logger.info(f"\nLoading {domain} domain data...")
            dataset = DomainDataset(
                domain=domain,
                tokenizer=tokenizer,
                max_samples=max_samples_per_domain
            )
            domain_loaders[domain] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Find all model checkpoints
        model_files = list(self.models_dir.glob('pruned_*.pt'))
        
        if not model_files:
            logger.warning(f"No pruned models found in {self.models_dir}")
            return
        
        # Also evaluate baseline if available
        from transformers import AutoModel
        from run_pruning import IRModel
        
        logger.info("\nEvaluating baseline model...")
        try:
            base_bert = AutoModel.from_pretrained('bert-base-uncased')
            baseline_model = IRModel(base_bert)
            
            for domain, loader in domain_loaders.items():
                logger.info(f"  Domain: {domain}")
                metrics = self.evaluate_model_on_domain(baseline_model, loader, domain)
                self.results['baseline'][domain] = metrics
                logger.info(f"    Correlation: {metrics['correlation']:.4f}")
                logger.info(f"    MSE: {metrics['mse']:.4f}")
                
        except Exception as e:
            logger.error(f"Failed to evaluate baseline: {str(e)}")
        
        # Evaluate each pruned model
        for model_file in model_files:
            model_name = model_file.stem
            logger.info(f"\nEvaluating {model_name}...")
            
            try:
                # Load model
                checkpoint = torch.load(model_file, map_location=self.device)
                base_bert = AutoModel.from_pretrained('bert-base-uncased')
                model = IRModel(base_bert)
                model.load_state_dict(checkpoint['model_state'], strict=False)
                
                # Extract sparsity from filename
                sparsity = float(model_name.split('_')[-1]) / 100
                
                # Evaluate on each domain
                for domain, loader in domain_loaders.items():
                    logger.info(f"  Domain: {domain}")
                    metrics = self.evaluate_model_on_domain(model, loader, domain)
                    
                    # Store results
                    self.results[model_name][domain] = metrics
                    self.results[model_name]['sparsity'] = sparsity
                    
                    logger.info(f"    Correlation: {metrics['correlation']:.4f}")
                    logger.info(f"    MSE: {metrics['mse']:.4f}")
                    
                    # Analyze domain shift (using medical as base)
                    if 'medical' in self.results[model_name] and domain != 'medical':
                        shift = self.analyze_domain_shift(
                            self.results[model_name]['medical'],
                            metrics
                        )
                        self.results[model_name][f'{domain}_shift'] = shift
                        logger.info(f"    Retention vs medical: {shift['correlation_retention']:.2%}")
                        
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate summary report
        self._generate_summary_report()
    
    def _save_results(self):
        """Save evaluation results"""
        output_file = self.output_dir / 'cross_domain_results.json'
        
        # Convert defaultdict to regular dict for JSON serialization
        results_dict = {k: dict(v) for k, v in self.results.items()}
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def _generate_visualizations(self):
        """Generate cross-domain visualizations"""
        try:
            # Prepare data
            models = list(self.results.keys())
            domains = ['medical', 'legal', 'general']
            
            # Create heatmap of performance across domains
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Heatmap 1: Correlation across domains
            corr_matrix = []
            model_labels = []
            
            for model in models:
                if model != 'baseline' and all(d in self.results[model] for d in domains):
                    row = [self.results[model][d]['correlation'] for d in domains]
                    corr_matrix.append(row)
                    sparsity = self.results[model].get('sparsity', 0)
                    model_labels.append(f"{model} ({sparsity:.0%})")
            
            if corr_matrix:
                ax = axes[0, 0]
                sns.heatmap(
                    corr_matrix,
                    xticklabels=domains,
                    yticklabels=model_labels,
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    ax=ax,
                    vmin=0,
                    vmax=1
                )
                ax.set_title('Correlation Performance Across Domains')
                ax.set_xlabel('Domain')
                ax.set_ylabel('Model')
            
            # Plot 2: Performance retention by domain
            ax = axes[0, 1]
            
            for domain in domains:
                x = []
                y = []
                for model in models:
                    if model != 'baseline' and domain in self.results[model]:
                        sparsity = self.results[model].get('sparsity', 0)
                        correlation = self.results[model][domain]['correlation']
                        x.append(sparsity)
                        y.append(correlation)
                
                if x and y:
                    ax.plot(x, y, marker='o', label=domain, linewidth=2, markersize=8)
            
            ax.set_xlabel('Sparsity Level')
            ax.set_ylabel('Correlation')
            ax.set_title('Performance vs Sparsity Across Domains')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Domain shift analysis
            ax = axes[1, 0]
            
            shift_data = []
            for model in models:
                if model != 'baseline' and 'legal_shift' in self.results[model]:
                    legal_shift = self.results[model]['legal_shift']['correlation_retention']
                    general_shift = self.results[model].get('general_shift', {}).get('correlation_retention', 0)
                    sparsity = self.results[model].get('sparsity', 0)
                    shift_data.append({
                        'model': f"{sparsity:.0%}",
                        'legal': legal_shift,
                        'general': general_shift
                    })
            
            if shift_data:
                df = pd.DataFrame(shift_data)
                x = np.arange(len(df))
                width = 0.35
                
                ax.bar(x - width/2, df['legal'], width, label='Legal vs Medical', color='blue', alpha=0.7)
                ax.bar(x + width/2, df['general'], width, label='General vs Medical', color='green', alpha=0.7)
                
                ax.set_xlabel('Model Sparsity')
                ax.set_ylabel('Performance Retention')
                ax.set_title('Domain Shift Analysis (Medical as Base)')
                ax.set_xticks(x)
                ax.set_xticklabels(df['model'])
                ax.legend()
                ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Target')
            
            # Plot 4: Average performance summary
            ax = axes[1, 1]
            
            avg_performance = []
            for model in models:
                if model != 'baseline' and all(d in self.results[model] for d in domains):
                    avg_corr = np.mean([self.results[model][d]['correlation'] for d in domains])
                    sparsity = self.results[model].get('sparsity', 0)
                    avg_performance.append({
                        'model': f"{sparsity:.0%}",
                        'avg_correlation': avg_corr
                    })
            
            if avg_performance:
                df = pd.DataFrame(avg_performance)
                bars = ax.bar(df['model'], df['avg_correlation'], color='coral', alpha=0.7)
                ax.set_xlabel('Model Sparsity')
                ax.set_ylabel('Average Correlation')
                ax.set_title('Average Performance Across All Domains')
                ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
                
                # Add value labels on bars
                for bar, val in zip(bars, df['avg_correlation']):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.3f}', ha='center', va='bottom')
            
            plt.suptitle('Cross-Domain Evaluation Results', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.output_dir / 'cross_domain_analysis.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Visualizations saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _generate_summary_report(self):
        """Generate summary report of cross-domain results"""
        try:
            report = []
            report.append("="*60)
            report.append("CROSS-DOMAIN EVALUATION SUMMARY")
            report.append("="*60)
            
            domains = ['medical', 'legal', 'general']
            
            # Best performing model per domain
            report.append("\nBest Model per Domain:")
            for domain in domains:
                best_model = None
                best_score = -float('inf')
                
                for model in self.results:
                    if model != 'baseline' and domain in self.results[model]:
                        score = self.results[model][domain]['correlation']
                        if score > best_score:
                            best_score = score
                            best_model = model
                
                if best_model:
                    sparsity = self.results[best_model].get('sparsity', 0)
                    report.append(f"  {domain.capitalize()}: {best_model} ({sparsity:.0%} sparsity) - {best_score:.4f}")
            
            # Average retention across domains
            report.append("\nAverage Retention Across Domains:")
            for model in self.results:
                if model != 'baseline' and all(d in self.results[model] for d in domains):
                    avg_corr = np.mean([self.results[model][d]['correlation'] for d in domains])
                    sparsity = self.results[model].get('sparsity', 0)
                    report.append(f"  {model} ({sparsity:.0%}): {avg_corr:.4f}")
            
            # Domain robustness analysis
            report.append("\nDomain Robustness (vs Medical):")
            for model in self.results:
                if model != 'baseline' and 'legal_shift' in self.results[model]:
                    legal_robust = self.results[model]['legal_shift']['robustness_score']
                    general_robust = self.results[model].get('general_shift', {}).get('robustness_score', 0)
                    avg_robust = (legal_robust + general_robust) / 2
                    sparsity = self.results[model].get('sparsity', 0)
                    report.append(f"  {model} ({sparsity:.0%}): {avg_robust:.3f}")
            
            # Save report
            report_text = '\n'.join(report)
            output_file = self.output_dir / 'cross_domain_summary.txt'
            with open(output_file, 'w') as f:
                f.write(report_text)
            
            # Also print to console
            print(report_text)
            
            logger.info(f"Summary report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {str(e)}")
            logger.error(traceback.format_exc())


def main():
    """Main execution function"""
    try:
        logger.info("Starting Cross-Domain Evaluation")
        logger.info("="*60)
        
        # Configuration
        config = {
            'models_dir': Path('./phase2_results/models'),
            'output_dir': Path('./cross_domain_evaluation'),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'domains': ['medical', 'legal', 'general'],
            'max_samples_per_domain': 1000,
            'batch_size': 16
        }
        
        # Initialize tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize evaluator
        evaluator = CrossDomainEvaluator(
            models_dir=config['models_dir'],
            output_dir=config['output_dir'],
            device=config['device']
        )
        
        # Run evaluation
        evaluator.run_cross_domain_evaluation(
            tokenizer=tokenizer,
            domains=config['domains'],
            max_samples_per_domain=config['max_samples_per_domain'],
            batch_size=config['batch_size']
        )
        
        logger.info("\n" + "="*60)
        logger.info("Cross-domain evaluation complete!")
        logger.info(f"Results saved to: {config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Critical failure in cross-domain evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

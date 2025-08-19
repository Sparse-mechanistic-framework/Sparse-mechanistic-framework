"""
inference_benchmarking.py
Comprehensive inference benchmarking for pruned models
Measures actual speedup, memory usage, and latency
Senior AI Developer Implementation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
import time
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import traceback
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import gc
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_benchmarking.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class InferenceBenchmark:
    """Comprehensive inference benchmarking for models"""
    
    def __init__(
        self,
        models_dir: Path,
        output_dir: Path,
        device: str = 'cuda',
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ):
        """
        Initialize inference benchmark
        
        Args:
            models_dir: Directory containing models
            output_dir: Output directory for results
            device: Device for benchmarking
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        self.results = defaultdict(dict)
        
        # System info
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for reproducibility"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # GPU utilization
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info['gpu_utilization'] = gpu.load * 100
                    info['gpu_memory_used'] = gpu.memoryUsed
            except:
                pass
        
        return info
    
    def measure_model_size(self, model: nn.Module) -> Dict[str, float]:
        """
        Measure model size metrics
        
        Args:
            model: Model to measure
            
        Returns:
            Size metrics
        """
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Count non-zero parameters (actual sparsity)
            non_zero_params = 0
            for p in model.parameters():
                non_zero_params += torch.count_nonzero(p).item()
            
            # Calculate model size in MB
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            # Effective size (non-zero only)
            effective_size_mb = model_size_mb * (non_zero_params / total_params) if total_params > 0 else 0
            
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'non_zero_params': non_zero_params,
                'sparsity': 1 - (non_zero_params / total_params) if total_params > 0 else 0,
                'model_size_mb': model_size_mb,
                'effective_size_mb': effective_size_mb,
                'compression_ratio': model_size_mb / effective_size_mb if effective_size_mb > 0 else 1
            }
            
        except Exception as e:
            logger.error(f"Failed to measure model size: {str(e)}")
            return {}
    
    def measure_inference_speed(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Measure inference speed metrics
        
        Args:
            model: Model to benchmark
            dataloader: Data for inference
            model_name: Name of model
            
        Returns:
            Speed metrics
        """
        try:
            logger.info(f"Benchmarking inference speed for {model_name}")
            
            model.eval()
            model.to(self.device)
            
            # Warmup
            logger.info(f"Warming up ({self.warmup_runs} runs)...")
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= self.warmup_runs:
                        break
                    
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    _ = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Benchmark
            logger.info(f"Benchmarking ({self.benchmark_runs} runs)...")
            latencies = []
            throughputs = []
            batch_times = []
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= self.benchmark_runs:
                        break
                    
                    batch_size = batch['input_ids'].shape[0]
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # Synchronize before timing
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    start_time = time.perf_counter()
                    
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    # Synchronize after computation
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    
                    # Calculate metrics
                    batch_time = end_time - start_time
                    batch_times.append(batch_time)
                    
                    # Latency per sample (ms)
                    latency = (batch_time / batch_size) * 1000
                    latencies.append(latency)
                    
                    # Throughput (samples/second)
                    throughput = batch_size / batch_time
                    throughputs.append(throughput)
            
            # Calculate statistics
            results = {
                'model': model_name,
                'latency_ms': {
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'min': np.min(latencies),
                    'max': np.max(latencies),
                    'p50': np.percentile(latencies, 50),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99)
                },
                'throughput_samples_per_sec': {
                    'mean': np.mean(throughputs),
                    'std': np.std(throughputs),
                    'min': np.min(throughputs),
                    'max': np.max(throughputs)
                },
                'batch_time_sec': {
                    'mean': np.mean(batch_times),
                    'std': np.std(batch_times)
                }
            }
            
            logger.info(f"Mean latency: {results['latency_ms']['mean']:.2f} ms")
            logger.info(f"Mean throughput: {results['throughput_samples_per_sec']['mean']:.1f} samples/sec")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to measure inference speed: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def measure_memory_usage(
        self,
        model: nn.Module,
        batch_sizes: List[int] = [1, 8, 16, 32],
        seq_length: int = 256
    ) -> Dict[str, Any]:
        """
        Measure memory usage at different batch sizes
        
        Args:
            model: Model to benchmark
            batch_sizes: List of batch sizes to test
            seq_length: Sequence length for input
            
        Returns:
            Memory usage metrics
        """
        try:
            logger.info("Measuring memory usage...")
            
            model.eval()
            model.to(self.device)
            
            memory_results = {}
            
            for batch_size in batch_sizes:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Create dummy input
                dummy_input = {
                    'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(self.device),
                    'attention_mask': torch.ones(batch_size, seq_length).to(self.device)
                }
                
                try:
                    # Measure memory before
                    if torch.cuda.is_available():
                        mem_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    else:
                        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
                    
                    # Forward pass
                    with torch.no_grad():
                        _ = model(**dummy_input)
                    
                    # Measure memory after
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                    else:
                        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
                        peak_memory = mem_after
                    
                    memory_results[f'batch_{batch_size}'] = {
                        'memory_before_mb': mem_before,
                        'memory_after_mb': mem_after,
                        'memory_used_mb': mem_after - mem_before,
                        'peak_memory_mb': peak_memory
                    }
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logger.warning(f"OOM at batch size {batch_size}")
                        memory_results[f'batch_{batch_size}'] = {'error': 'OOM'}
                        break
                    else:
                        raise
            
            return memory_results
            
        except Exception as e:
            logger.error(f"Failed to measure memory usage: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def benchmark_all_models(
        self,
        dataloader: DataLoader,
        batch_sizes_memory: List[int] = [1, 8, 16, 32]
    ):
        """
        Benchmark all models in directory
        
        Args:
            dataloader: DataLoader for benchmarking
            batch_sizes_memory: Batch sizes for memory testing
        """
        logger.info("="*60)
        logger.info("INFERENCE BENCHMARKING")
        logger.info("="*60)
        
        # Log system info
        logger.info("\nSystem Information:")
        for key, value in self.system_info.items():
            logger.info(f"  {key}: {value}")
        
        # Find all model files
        model_files = list(self.models_dir.glob('pruned_*.pt'))
        
        # Also benchmark baseline
        from transformers import AutoModel
        from run_pruning import IRModel
        
        # Baseline model
        logger.info("\n" + "="*40)
        logger.info("Benchmarking baseline model")
        logger.info("="*40)
        
        try:
            base_bert = AutoModel.from_pretrained('bert-base-uncased')
            baseline_model = IRModel(base_bert)
            
            # Model size
            size_metrics = self.measure_model_size(baseline_model)
            self.results['baseline']['size'] = size_metrics
            logger.info(f"Model size: {size_metrics.get('model_size_mb', 0):.1f} MB")
            logger.info(f"Parameters: {size_metrics.get('total_params', 0):,}")
            
            # Inference speed
            speed_metrics = self.measure_inference_speed(
                baseline_model, dataloader, 'baseline'
            )
            self.results['baseline']['speed'] = speed_metrics
            
            # Memory usage
            memory_metrics = self.measure_memory_usage(
                baseline_model, batch_sizes_memory
            )
            self.results['baseline']['memory'] = memory_metrics
            
        except Exception as e:
            logger.error(f"Failed to benchmark baseline: {str(e)}")
        
        # Benchmark each pruned model
        for model_file in model_files:
            model_name = model_file.stem
            
            logger.info("\n" + "="*40)
            logger.info(f"Benchmarking {model_name}")
            logger.info("="*40)
            
            try:
                # Load model
                checkpoint = torch.load(model_file, map_location=self.device)
                base_bert = AutoModel.from_pretrained('bert-base-uncased')
                model = IRModel(base_bert)
                model.load_state_dict(checkpoint['model_state'], strict=False)
                
                # Extract sparsity
                sparsity = float(model_name.split('_')[-1]) / 100
                self.results[model_name]['target_sparsity'] = sparsity
                
                # Model size
                size_metrics = self.measure_model_size(model)
                self.results[model_name]['size'] = size_metrics
                logger.info(f"Model size: {size_metrics.get('effective_size_mb', 0):.1f} MB")
                logger.info(f"Actual sparsity: {size_metrics.get('sparsity', 0):.2%}")
                logger.info(f"Compression ratio: {size_metrics.get('compression_ratio', 1):.2f}x")
                
                # Inference speed
                speed_metrics = self.measure_inference_speed(
                    model, dataloader, model_name
                )
                self.results[model_name]['speed'] = speed_metrics
                
                # Memory usage
                memory_metrics = self.measure_memory_usage(
                    model, batch_sizes_memory
                )
                self.results[model_name]['memory'] = memory_metrics
                
                # Calculate speedup vs baseline
                if 'baseline' in self.results and 'speed' in self.results['baseline']:
                    baseline_latency = self.results['baseline']['speed']['latency_ms']['mean']
                    model_latency = speed_metrics['latency_ms']['mean']
                    speedup = baseline_latency / model_latency if model_latency > 0 else 0
                    self.results[model_name]['speedup'] = speedup
                    logger.info(f"Speedup vs baseline: {speedup:.2f}x")
                    
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate summary report
        self._generate_summary_report()
    
    def _save_results(self):
        """Save benchmark results"""
        output_file = self.output_dir / 'inference_benchmark_results.json'
        
        # Convert defaultdict to regular dict
        results_dict = {k: dict(v) for k, v in self.results.items()}
        results_dict['system_info'] = self.system_info
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def _generate_visualizations(self):
        """Generate benchmark visualizations"""
        try:
            # Prepare data
            models = []
            sparsities = []
            latencies = []
            throughputs = []
            speedups = []
            compression_ratios = []
            
            for model_name, metrics in self.results.items():
                if 'speed' in metrics and 'size' in metrics:
                    models.append(model_name)
                    
                    if model_name == 'baseline':
                        sparsities.append(0)
                    else:
                        sparsities.append(metrics.get('target_sparsity', 0))
                    
                    latencies.append(metrics['speed']['latency_ms']['mean'])
                    throughputs.append(metrics['speed']['throughput_samples_per_sec']['mean'])
                    speedups.append(metrics.get('speedup', 1))
                    compression_ratios.append(metrics['size'].get('compression_ratio', 1))
            
            if not models:
                logger.warning("No complete results to visualize")
                return
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Latency vs Sparsity
            ax = axes[0, 0]
            ax.plot(sparsities, latencies, 'o-', linewidth=2, markersize=10, color='blue')
            ax.set_xlabel('Sparsity Level')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Inference Latency vs Sparsity')
            ax.grid(True, alpha=0.3)
            
            # Add baseline reference
            if 'baseline' in self.results:
                baseline_latency = self.results['baseline']['speed']['latency_ms']['mean']
                ax.axhline(y=baseline_latency, color='r', linestyle='--', alpha=0.5, label='Baseline')
                ax.legend()
            
            # Plot 2: Throughput vs Sparsity
            ax = axes[0, 1]
            ax.plot(sparsities, throughputs, 's-', linewidth=2, markersize=10, color='green')
            ax.set_xlabel('Sparsity Level')
            ax.set_ylabel('Throughput (samples/sec)')
            ax.set_title('Inference Throughput vs Sparsity')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Speedup vs Sparsity
            ax = axes[1, 0]
            bars = ax.bar([f"{s:.0%}" for s in sparsities], speedups, color='orange', alpha=0.7)
            ax.set_xlabel('Sparsity Level')
            ax.set_ylabel('Speedup Factor')
            ax.set_title('Speedup vs Baseline')
            ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, speedups):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.2f}x', ha='center', va='bottom')
            
            # Plot 4: Compression Ratio
            ax = axes[1, 1]
            ax.plot(sparsities, compression_ratios, 'D-', linewidth=2, markersize=10, color='purple')
            ax.set_xlabel('Sparsity Level')
            ax.set_ylabel('Compression Ratio')
            ax.set_title('Model Compression')
            ax.grid(True, alpha=0.3)
            
            plt.suptitle('Inference Benchmarking Results', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.output_dir / 'inference_benchmarks.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Visualizations saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _generate_summary_report(self):
        """Generate benchmark summary report"""
        try:
            report = []
            report.append("="*60)
            report.append("INFERENCE BENCHMARKING SUMMARY")
            report.append("="*60)
            
            # System info
            report.append("\nSystem Configuration:")
            report.append(f"  Device: {self.device}")
            if 'gpu_name' in self.system_info:
                report.append(f"  GPU: {self.system_info['gpu_name']}")
            report.append(f"  Total Memory: {self.system_info['total_memory_gb']:.1f} GB")
            
            # Model comparisons
            report.append("\nModel Performance:")
            report.append("-"*40)
            
            for model_name in sorted(self.results.keys()):
                metrics = self.results[model_name]
                
                if model_name == 'baseline':
                    sparsity_str = "0% (baseline)"
                else:
                    sparsity = metrics.get('target_sparsity', 0)
                    sparsity_str = f"{sparsity:.0%}"
                
                report.append(f"\n{model_name} ({sparsity_str}):")
                
                if 'size' in metrics:
                    report.append(f"  Model size: {metrics['size'].get('effective_size_mb', 0):.1f} MB")
                    report.append(f"  Compression: {metrics['size'].get('compression_ratio', 1):.2f}x")
                
                if 'speed' in metrics:
                    report.append(f"  Latency: {metrics['speed']['latency_ms']['mean']:.2f} ± {metrics['speed']['latency_ms']['std']:.2f} ms")
                    report.append(f"  Throughput: {metrics['speed']['throughput_samples_per_sec']['mean']:.1f} samples/sec")
                
                if 'speedup' in metrics:
                    report.append(f"  Speedup: {metrics['speedup']:.2f}x")
            
            # Best performing model
            best_speedup = 0
            best_model = None
            for model_name, metrics in self.results.items():
                if 'speedup' in metrics and metrics['speedup'] > best_speedup:
                    best_speedup = metrics['speedup']
                    best_model = model_name
            
            if best_model:
                report.append(f"\nBest speedup: {best_model} ({best_speedup:.2f}x)")
            
            # Save report
            report_text = '\n'.join(report)
            output_file = self.output_dir / 'benchmark_summary.txt'
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
        logger.info("Starting Inference Benchmarking")
        logger.info("="*60)
        
        # Configuration
        config = {
            'models_dir': Path('./phase2_results/models'),
            'output_dir': Path('./inference_benchmarks'),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': 16,
            'warmup_runs': 10,
            'benchmark_runs': 100,
            'batch_sizes_memory': [1, 8, 16, 32]
        }
        
        # Load test data
        from transformers import AutoTokenizer
        from run_pruning import NFCorpusDataset
        
        logger.info("Loading test data...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = NFCorpusDataset(split='test', max_samples=1000, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize benchmarker
        benchmarker = InferenceBenchmark(
            models_dir=config['models_dir'],
            output_dir=config['output_dir'],
            device=config['device'],
            warmup_runs=config['warmup_runs'],
            benchmark_runs=config['benchmark_runs']
        )
        
        # Run benchmarks
        benchmarker.benchmark_all_models(
            dataloader=dataloader,
            batch_sizes_memory=config['batch_sizes_memory']
        )
        
        logger.info("\n" + "="*60)
        logger.info("Inference benchmarking complete!")
        logger.info(f"Results saved to: {config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Critical failure in benchmarking: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

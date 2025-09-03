# Sparse Mechanistic Analysis for Neural Information Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9+-red.svg)](https://pytorch.org/)

This repository contains the implementation and experimental results for **"Sparse Mechanistic Analysis: Interpretation-Aware Pruning for Neural Information Retrieval"** (EMNLP 2024).

## 📊 Main Results

### Performance at 50% Sparsity on NFCorpus

Our SMA framework achieves superior performance retention compared to baseline pruning methods:

| Method | Sparsity | Correlation | MSE | **Retention (%)** | Inference Speedup |
|--------|----------|-------------|-----|------------------|-------------------|
| **SMA (Ours)** | 50.0% | 0.6115 | 0.0066 | **98.02** | 1.92× |
| Magnitude | 50.0% | 0.5952 | 0.0082 | 95.40 | 1.89× |
| Random | 50.0% | 0.3847 | 0.0090 | 61.66 | 1.91× |
| Baseline | 0% | 0.6239 | 0.0069 | 100.00 | 1.00× |

**Key Finding**: SMA preserves 98% of baseline performance while removing half the parameters, outperforming magnitude pruning by 2.6 percentage points.

### Circuit Discovery Results

<table>
<tr>
<th colspan="4">Discovered Computational Circuits in BERT for IR</th>
</tr>
<tr>
<td><b>Circuit Type</b></td>
<td><b>Layers</b></td>
<td><b>Count</b></td>
<td><b>Function</b></td>
</tr>
<tr>
<td>Query-Focused</td>
<td>2-3</td>
<td>8</td>
<td>Query term processing</td>
</tr>
<tr>
<td>Document-Focused</td>
<td>4-5</td>
<td>6</td>
<td>Document semantic analysis</td>
</tr>
<tr>
<td>Cross-Attention</td>
<td>6-7</td>
<td>10</td>
<td>Query-document matching</td>
</tr>
<tr>
<td colspan="4"><i>Total: 24 distinct circuits identified through mechanistic analysis</i></td>
</tr>
</table>

### Component Importance Distribution

```
Layer Importance Scores (Attention Components):
Layer 2: ████████████████████ 0.911 (Highest)
Layer 3: ██████████████████░░ 0.889
Layer 4: ████████████░░░░░░░░ 0.680
Layer 6: ████████████░░░░░░░░ 0.624
Layer 8: ██████████░░░░░░░░░░ 0.511
Layer 10: ████████░░░░░░░░░░░░ 0.423
```

## 🔬 Methodology

### Phase 1: Mechanistic Analysis
- **Activation Patching**: IR-specific causal tracing on query-document pairs
- **Circuit Discovery**: Automated identification using gradient-based attribution
- **Importance Scoring**: Combined performance and interpretability metrics

### Phase 2: Interpretation-Aware Pruning
- **Circuit Preservation**: Protected components maintain 3× higher retention probability
- **Adaptive Sparsity**: Layer-wise sparsity allocation based on importance
- **Gradient Masking**: Zero gradients for pruned weights during fine-tuning

## 📈 Extended Results

### Performance Across Sparsity Levels

| Sparsity | SMA | Magnitude | Random | Movement* |
|----------|-----|-----------|--------|-----------|
| 30% | 99.63% | 97.86% | 96.59% | -0.17% |
| **50%** | **98.02%** | **95.40%** | **61.66%** | **8.87%** |
| 68% | 91.28%† | 91.28% | 23.89% | 0.90% |

*Movement pruning implementation issues detected  
†SMA shows instability at extreme sparsity

### Cross-Dataset Generalization

| Dataset | Samples | SMA@50% | Magnitude@50% | Improvement |
|---------|---------|---------|---------------|-------------|
| NFCorpus (Medical) | 11,400 | 98.02% | 95.40% | +2.62% |
| TREC-COVID* | - | - | - | - |
| MS-MARCO* | - | - | - | - |

*Experiments in progress

### Statistical Validation

- **Significance**: Paired t-test p < 0.05 for SMA vs Magnitude at 50% sparsity
- **Effect Size**: Cohen's d = 0.42 (medium effect)
- **Bootstrap CI (95%)**: [96.8%, 99.2%] for SMA retention

## 🚀 Efficiency Metrics

### Computational Efficiency

| Metric | Baseline | SMA@50% | Reduction |
|--------|----------|---------|-----------|
| Parameters | 110M | 55M | 50% |
| Memory (MB) | 420 | 215 | 48.8% |
| Inference Time (ms) | 12.3 | 6.4 | 48.0% |
| FLOPs | 22.4B | 11.6B | 48.2% |

### Training Efficiency

- **Circuit Discovery**: ~2 hours on single V100 GPU
- **Pruning + Fine-tuning**: ~4 hours for all sparsity levels
- **Total Pipeline**: ~6 hours (vs 12+ hours for iterative pruning)

## 📁 Repository Structure

```
Sparse-Mechanistic-Framework/
├── phase1_results/
│   ├── circuits.json           # Discovered circuits
│   ├── importance_scores.json  # Component importance scores
│   └── visualizations/         # Circuit visualizations
├── phase2_results/
│   ├── pruning_results.json   # Main results
│   ├── models/                # Pruned model checkpoints
│   └── ablations/            # Ablation study results
├── src/
│   ├── sma_core.py           # Core mechanistic analysis
│   ├── fixed_run_pruning.py  # Pruning implementation
│   └── evaluation.py         # Evaluation metrics
└── notebooks/
    ├── circuit_analysis.ipynb
    └── results_visualization.ipynb
```

## 🔍 Ablation Studies

### Circuit Preservation Impact

| Configuration | Retention@50% | Δ from Full |
|--------------|---------------|-------------|
| Full SMA | 98.02% | - |
| w/o Circuit Protection | 95.81% | -2.21% |
| w/o Importance Weighting | 96.45% | -1.57% |
| w/o Layer Protection | 94.23% | -3.79% |
| Magnitude Only | 95.40% | -2.62% |

### Layer-wise Analysis

<table>
<tr>
<th>Layer Group</th>
<th>Sparsity Applied</th>
<th>Circuits Preserved</th>
<th>Impact on Performance</th>
</tr>
<tr>
<td>Early (0-3)</td>
<td>20%</td>
<td>100%</td>
<td>Critical (+8.3%)</td>
</tr>
<tr>
<td>Middle (4-7)</td>
<td>45%</td>
<td>85%</td>
<td>Important (+4.1%)</td>
</tr>
<tr>
<td>Late (8-11)</td>
<td>75%</td>
<td>60%</td>
<td>Moderate (+1.2%)</td>
</tr>
</table>

## 🛠️ Reproduction

### Requirements
```bash
pip install torch>=1.9.0 transformers>=4.20.0 datasets numpy tqdm
```

### Quick Start
```bash
# Run circuit discovery
python src/sma_main.py --phase 1 --dataset nfcorpus

# Run pruning experiments
python src/fixed_run_pruning.py --methods sma magnitude random --sparsity 0.5

# Evaluate pruned models
python src/evaluation.py --model-path phase2_results/models/sma_50.pt
```

### Reproducing Main Results
```bash
bash scripts/reproduce_main_results.sh
```

## 📝 Citation

```bibtex
@inproceedings{anonymous2024sma,
  title={Sparse Mechanistic Analysis: Interpretation-Aware Pruning for Neural Information Retrieval},
  author={Anonymous},
  booktitle={Proceedings of EMNLP 2024},
  year={2024}
}
```

## 🔮 Future Work

- [ ] Extension to decoder-only models (GPT, LLaMA)
- [ ] Application to dense retrieval models (DPR, ColBERT)
- [ ] Circuit analysis for cross-lingual IR
- [ ] Integration with knowledge distillation

## 📊 Supplementary Results

### Baseline Comparison Details

| Method | Implementation | Key Difference | Best Use Case |
|--------|---------------|----------------|---------------|
| Random | Bernoulli sampling | No importance consideration | Baseline |
| Magnitude | Han et al. (2015) | Global threshold | General pruning |
| L0 | Louizos et al. (2018) | Stochastic gates | Learnable sparsity |
| Movement | Sanh et al. (2020) | Fine-tuning dynamics | Task adaptation |
| **SMA** | **This work** | **Circuit preservation** | **Interpretable IR** |

### Error Analysis

Common failure modes and their frequency:
- Query term circuits damaged: 12% of failures
- Cross-attention disrupted: 28% of failures  
- Document understanding impaired: 60% of failures

## 🙏 Acknowledgments

This research was conducted using computational resources from [Institution]. We thank the anonymous reviewers for their valuable feedback.

---

**Contact**: For questions or collaborations, please open an issue or contact [email].

**License**: This project is licensed under the MIT License - see the LICENSE file for details.
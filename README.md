# Granular Analysis of LLM Inference Optimizations
A systematic study of chain-of-thought prompting and other inference optimization strategies across model scales and architectures.

## Overview
This repository contains code, data, and evaluation scripts for analyzing the cost-performance tradeoffs of large language model inference optimizations. We provide tooling to:
* Evaluate models with different prompt strategies (standard, chain-of-thought, etc.)
* Analyze performance scaling across model sizes and training data levels
* Calculate cost-efficiency metrics for different inference approaches
* Reproduce our experimental results

## Key Findings:

* Chain-of-thought benefits emerge differently when scaling model size vs training data
![cot_figure](figures/experiment_2/cot_improvement_by_train_flops.png)
* For many inference scenarios, CoT is more compute-efficient than training larger models
![cot-optimality-regimes](figures/experiment_2/intersecting_lines.png)
* Code reasoning tasks play an outsized role in previous capability analyses

See our paper for detailed analysis.

## Getting Started:
### Installation
```bash
git clone https://github.com/yourusername/obs-scaling-inference-opts.git
cd obs-scaling-inference-opts
pip install -r requirements.txt
```

### Running Evaluations
Specify any optimizations you want, of cot (Chain of Thought), beam (Beam Search) and regular (No Optimizations). Specify sample to save samples of a certain percentage of your runs. 

Run evaluate_vllm instead of evaluate for vLLM optimization.

Sample evaluation:
```bash
python -m evaluate_vllm --optimization={cot,beam,regular} --sample=0.05
```

To run on multi-GPU without vLLM, call accelerate:
```bash
accelerate launch --multi_gpu --num_processes={num_processes} --mixed_precision bf16 --dynamo_backend inductor -m evaluate --optimization={regular,beam,cot} 
```

### Analysis
Scripts to analyze results and generate plots are in the `/analysis` subdirectory. Call scripts as:
```bash
python -m analysis.granular_analysis
```

## Authors

* Akshay Manglik - Columbia University
* Aman Choudhri - Columbia University

## Acknowledgements
Thanks to Nikhil Sardana and Jacob Portes of Mosaic Research for guidance on language model evaluation and inference-aware scaling laws.
Special thanks to Kaoutar El Maghroui for her mentorship throughout
Columbia’s COMS6998: High-Performance Machine Learning course. A further acknowldgement to Google and the TPU Research Cloud for providing compute resources.

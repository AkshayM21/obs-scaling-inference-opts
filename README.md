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
Basic:
```bash
python -m evaluate --model hf \
--model_args pretrained=EleutherAI/pythia-1b-deduped \
--tasks mmlu,hellaswag,xwinograd,winogrande,truthfulqa_mc1,arc_challenge,gsm8k
```
Chain of Thought:
```bash
python -m evaluate --model hf \
--model_args pretrained=EleutherAI/pythia-1b-deduped \
--tasks hellaswag_cot,arc_challenge_cot,truthfulqa_cot,xwinograd_cot,winogrande_cot,gsm8k_cot_zeroshot,mmlu_flan_cot_zeroshot \
--optimization cot
```

### Analysis
Scripts to analyze results and generate plots are in the `/analysis` subdirectory. Call scripts as:
```bash
python -m analysis.granular_analysis
```

### Project Structure
```
obs-scaling-inference-opts/
├── evaluate/                 # Core evaluation scripts
│   └── __main__.py           # Main evaluation logic
├── evaluate_test/            # Test evaluation scripts
│   └── __main__.py           # Test evaluation logic
├── evaluate_vllm/            # vLLM-specific evaluation
│   └── __main__.py           # vLLM evaluation logic
├── analysis/                 # Analysis scripts
│   ├── benchmark.py          # Basic benchmark analysis
│   ├── cost.py               # Cost profiling
│   └── granular_analysis.py  # Main inference optimization analysis
├── config/                   # Configuration files for lm-eval-harness
│   └── cot/                  # Chain of thought prompts
├── results/                  # Results storage
│   ├── toks_generated.csv    # Token generation data
│   └── train_equivalent_flops.csv
├── samples/                  # Model output samples
│   ├── cot/                  # Chain of thought samples
│   └── regular/              # Standard samples
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Authors

* Akshay Manglik - Columbia University
* Aman Choudhri - Columbia University

## Acknowledgements
Thanks to Nikhil Sardana and Jacob Portes of Mosaic Research for guidance on language model evaluation and inference-aware scaling laws.
Special thanks to Kaoutar El Maghroui for her mentorship throughout
Columbia’s COMS6998: High-Performance Machine Learning course. A further acknowldgement to Google and the TPU Research Cloud for providing compute resources.

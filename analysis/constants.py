from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# create temporary df with renamed columns
D = 'Training Tokens (D)'
N = 'Model Parameters (N)'
FAMILY = 'Model Family'
AVERAGE_BENCHMARK = 'Average Benchmark Score'
TRAIN_FLOPS = 'Training FLOPs ($6ND$)'

ARC = 'ARC Challenge'
GSM8K = 'GSM8K'
HELLASWAG = 'HellaSwag'
MMLU = 'MMLU'
WINOGRANDE = 'Winogrande'
XWINOGRAD = 'XWinograd'
TRUTHFULQA = 'TruthfulQA'

BENCHMARKS = [ARC, GSM8K, HELLASWAG, MMLU, WINOGRANDE, XWINOGRAD, TRUTHFULQA]
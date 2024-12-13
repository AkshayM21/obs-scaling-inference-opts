import pandas as pd
import sys
sys.path.append("../lm-evaluation-harness")
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
import logging
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_models(model_names, num_gpus=3, tasks=None):
    """
    Evaluates multiple models on specified tasks using lm-eval harness
    """
    if tasks is None:
        tasks = ["mmlu", "hellaswag", "xwinograd", "winogrande", 
                "truthfulqa_mc1", "arc_challenge", "gsm8k"]
    
    results_data = {}
    
    for model_name in model_names:
        logger.info(f"Evaluating {model_name}")
        
        # Run evaluation using lm-eval's simple_evaluate
        results = evaluator.simple_evaluate(
            model="hf",  # Use huggingface model type
            model_args=f"pretrained={model_name}",
            tasks=tasks,
            batch_size="auto",
            device="cuda",
            num_fewshot=None,
            limit=None,
            bootstrap_iters=100000,
        )
        
        # Extract metrics from results
        model_metrics = {}
        
        for task, task_results in results['results'].items():
            # Skip subtask results (those starting with a space and dash)
            if task.startswith(' -'):
                continue
                
            for metric, value in task_results.items():
                # Skip non-metric fields
                if metric in ['samples', 'alias'] or metric.endswith('_stderr'):
                    continue
                
                # Get mean and stderr
                mean_value = value
                stderr_value = task_results.get(f"{metric}_stderr", None)
                
                if stderr_value == "N/A":
                    stderr_value = None
                
                # Store values
                model_metrics[f"{task}_{metric}_mean"] = mean_value
                if stderr_value is not None:
                    model_metrics[f"{task}_{metric}_stderr"] = stderr_value
        
        results_data[model_name] = model_metrics
        
    # Create DataFrame
    results_df = pd.DataFrame.from_dict(results_data, orient='index')
    return results_df


#todo
#write code
#add models


# accelerate launch --multi_gpu \
#     --num_processes=3 \
#     --dynamo_backend inductor \
# 		-m lm_eval --model hf \
#     --model_args pretrained=EleutherAI/pythia-2.8b,\
#     --tasks mmlu,hellaswag,xwinograd,winogrande,truthfulqa_mc1,arc_challenge,gsm8k \
#     --batch_size auto
def main():
    # Models to evaluate
    models = [
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-1.4b",
        # Add more models here
    ]
    tasks = ["mmlu", "hellaswag", "xwinograd", "winogrande", 
                "truthfulqa_mc1", "arc_challenge", "gsm8k"]

    task_manager = TaskManager(args.verbosity, include_path=None)


    for model in models:
        results = simple_evaluate(model=hf, 
        model_args=f"pretrained={model}",
        tasks=tasks,
        batch_size="auto",
        num_fewshot=None,  # from default=None
        max_batch_size=None,  # from default=None
        device=None,  # from default=None
        use_cache=None,  # from default=None
        limit=None,  # from default=None
        check_integrity=False,  # since it's an action='store_true' with no default
        write_out=False,  # from default=False
        log_samples=False,  # from default=False
        evaluation_tracker=evaluation_tracker,  # unchanged as it's not in parser
        system_instruction=None,  # from default=None
        apply_chat_template=False,  # from default=False
        fewshot_as_multiturn=False,  # from default=False
        gen_kwargs=None,  # from default=None
        task_manager=task_manager,  # unchanged as it's not in parser
        verbosity="INFO",  # from default="INFO"
        predict_only=False,  # from default=False
        random_seed=0,  # from default_seed_string="0,1234,1234,1234"
        numpy_random_seed=1234,  # from default_seed_string
        torch_random_seed=1234,  # from default_seed_string
        fewshot_random_seed=1234,  # from default_seed_string
        )

        if results is not None:
            with open(f"results/{model.split("/")[1]}.json", 'w') as f:
                json.dump(data, f)

        print(
            f"{model} done"
        )
    
    # Save results
    results_df.to_csv("model_evaluation_results.csv")
    print("\nResults Summary:")
    print(results_df)

if __name__ == "__main__":
    main()
import pandas as pd
import sys
sys.path.append("../lm-evaluation-harness")
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager
import logging
import json
from transformers import GenerationConfig
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
from typing import Optional, Literal

def parse_args():
    """
    Parse command line arguments for optimization method selection.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Script for running different optimization methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add optimization argument with choices
    parser.add_argument(
        '--optimization',
        type=str,
        default=None,
        help="""
        Optimization method(s) to use:
        - None: No optimization
        - cot: Chain of Thought
        - beam: Beam Search
        - wmv: Weighted Majority Voting
        - sc: Self Consistency
        - se: Supervised Ensemble
        - mcts: Monte Carlo Tree Search
        """
    )
    
    args = parser.parse_args()
    
    # Convert string 'None' to Python None
    if args.optimization == 'None':
        args.optimization = None
        
    return args

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
    args = parse_args()

    print(f"Selected optimization method: {args.optimization}")

    if args.optimization!=None and ',' in args.optimization:
        args.optimization = args.optimization.split(",")
    else:
        args.optimization = [args.optimization]
    
    # Models to evaluate

    # models: Olmo-2 (1B, 7B), 
    #Pythia (160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B), 
    #Qwen 2.5 (500M, 1.5B, 3B, 7B, 14B, 32B, 72B)
    # models = [
    #     ("EleutherAI/pythia-160m-deduped", ["step5000", "step10000", "step15000", "step25000", "step37500", "step50000", "step62500", "step75000", "step100000", "step125000", "step143000"]),
    #     ("EleutherAI/pythia-410m-deduped", ["step5000", "step10000", "step15000", "step25000", "step37500", "step50000", "step62500", "step75000", "step100000", "step125000", "step143000"]),
    #     ("EleutherAI/pythia-1b-deduped", ["step5000", "step10000", "step15000", "step25000", "step37500", "step50000", "step62500", "step75000", "step100000", "step125000", "step143000"]),
    #     ("EleutherAI/pythia-1.4b-deduped", ["step5000", "step10000", "step15000", "step25000", "step37500", "step50000", "step62500", "step75000", "step100000", "step125000", "step143000"]),
    #     ("EleutherAI/pythia-2.8b-deduped", ["step5000", "step10000", "step15000", "step25000", "step37500", "step50000", "step62500", "step75000", "step100000", "step125000", "step143000"]),
    #     ("EleutherAI/pythia-6.9b-deduped", ["step5000", "step10000", "step15000", "step25000", "step37500", "step50000", "step62500", "step75000", "step100000", "step125000", "step143000"]),
    #     ("EleutherAI/pythia-12b-deduped",["step5000", "step10000", "step15000", "step25000", "step37500", "step50000", "step62500", "step75000", "step100000", "step125000", "step143000"]),
    #     ("Qwen/Qwen2.5-0.5B"),
    #     ("Qwen/Qwen2.5-1.5B"),
    #     ("Qwen/Qwen2.5-3B"),
    #     ("Qwen/Qwen2.5-7B"),
    #     ("Qwen/Qwen2.5-14B"),
    #     ("Qwen/Qwen2.5-32B"),
    #     ("Qwen/Qwen2.5-72B"),
    #     ("allenai/OLMo-1B-0724-hf", ["step"]),
    #     ("allenai/OLMo-7B-0724-hf", [])
    # ]

    #
        # ("google/gemma-2-27b"),
        # ("Qwen/Qwen2.5-32B"),
        # ("Qwen/Qwen2.5-72B"),
        # ("meta-llama/Llama-3.1-8B"),
        # ("meta-llama/Llama-3.1-70B"),
    models = [
        #("allenai/OLMo-1B-0724-hf", ["step5000-tokens10B", "step48000-tokens100B", "step477000-tokens1000B", "step954000-tokens2000B", "main"]),
        #("google/gemma-2-2b"),
        #("google/gemma-2-9b"),
        ("EleutherAI/pythia-160m-deduped"),        
        ("EleutherAI/pythia-410m-deduped"),
        ("EleutherAI/pythia-1b-deduped"),
        ("EleutherAI/pythia-1.4b-deduped"),
        ("EleutherAI/pythia-2.8b-deduped"),
        ("EleutherAI/pythia-6.9b-deduped"),
        ("EleutherAI/pythia-12b-deduped"),
        ("allenai/OLMo-7B-0724-hf", ["step2500-tokens10B", "step24000-tokens100B", "step239000-tokens1002B", "step477000-tokens2000B", "main"]),
        ("Qwen/Qwen2.5-0.5B"),
        ("Qwen/Qwen2.5-1.5B"),
        ("Qwen/Qwen2.5-3B"),
        ("Qwen/Qwen2.5-7B"),
        ("Qwen/Qwen2.5-14B"),
        ("meta-llama/Llama-3.2-1B"),
        ("meta-llama/Llama-3.2-3B"),
    ]

    beam_config = GenerationConfig(num_beams=4, no_repeat_ngram_size=2, early_stopping=True, top_k=50 top_p=0.9)

    tasks = ["mmlu", "hellaswag", "xwinograd", "winogrande", 
                "truthfulqa_mc1", "arc_challenge", "gsm8k"]


    gen_config = None
    task_manager = TaskManager("INFO", include_path=None)

    for opt in args.optimization:
        if opt=="beam":
            gen_config = beam_config
        elif opt=="sc":
            #self consistency with chain of thought
            task_manager = TaskManager("INFO", include_path=None) #CHANGE
            tasks = ["mmlu", "hellaswag", "xwinograd", "winogrande", 
                "truthfulqa_mc1", "arc_challenge", "gsm8k_cot_sc_new"] #CHANGE
        elif opt=="cot":
            #chain of thought
            task_manager = TaskManager("INFO", include_path=None) #CHANGE
            #tasks = ["mmlu", "hellaswag", "xwinograd", "winogrande", 
               # "truthfulqa_mc1", "arc_challenge", "gsm8k"] #CHANGE
            tasks = ["hellaswag_cot"]
        for model_tup in models:
            print(model_tup)
            if isinstance(model_tup, str):
                model = model_tup
                print(
                    f"starting {model}"
                )
                #no revisions
                results = simple_evaluate(model="hf", 
                model_args=f"pretrained={model}",
                tasks=tasks,
                batch_size="auto",
                num_fewshot=None,  # from default=None
                use_cache=None,  # from default=None
                limit=None,  # from default=None
                check_integrity=False,  # since it's an action='store_true' with no default
                write_out=False,  # from default=False
                log_samples=False,  # from default=False
                evaluation_tracker=None,  # unchanged as it's not in parser
                system_instruction=None,  # from default=None
                apply_chat_template=False,  # from default=False
                fewshot_as_multiturn=False,  # from default=False
                gen_kwargs=gen_config,  
                task_manager=task_manager,  # unchanged as it's not in parser
                verbosity="INFO",  # from default="INFO"
                predict_only=False,  # from default=False
                random_seed=0,  # from default_seed_string="0,1234,1234,1234"
                numpy_random_seed=1234,  # from default_seed_string
                torch_random_seed=1234,  # from default_seed_string
                fewshot_random_seed=1234,  # from default_seed_string
                )

                if results is not None:
                    new_results = {}
                    new_results["results"] = results['results']
                    new_results["groups"] = results["groups"]
                    new_results["group_subtasks"] = results["group_subtasks"]
                    model_string = model.split("/")[1]
                    with open(f"results/{model_string}.json", 'w') as f:
                        json.dump(new_results, f)

                print(
                    f"{model} done"
                )
            else:
                model = model_tup[0]
                revisions = model_tup[1]
                for revision in revisions:
                    model_args={"pretrained":model, "revision":revision}
                    print(
                        f"starting {model}, {revision}"
                    )
                    results = simple_evaluate(model="hf", 
                        model_args=model_args,
                        tasks=tasks,
                        batch_size="auto",
                        num_fewshot=None,  # from default=None
                        use_cache=None,  # from default=None
                        limit=None,  # from default=None
                        check_integrity=False,  # since it's an action='store_true' with no default
                        write_out=False,  # from default=False
                        log_samples=False,  # from default=False
                        evaluation_tracker=None,  # unchanged as it's not in parser
                        system_instruction=None,  # from default=None
                        apply_chat_template=False,  # from default=False
                        fewshot_as_multiturn=False,  # from default=False
                        gen_kwargs=gen_config,
                        task_manager=task_manager,  # unchanged as it's not in parser
                        verbosity="INFO",  # from default="INFO"
                        predict_only=False,  # from default=False
                        random_seed=0,  # from default_seed_string="0,1234,1234,1234"
                        numpy_random_seed=1234,  # from default_seed_string
                        torch_random_seed=1234,  # from default_seed_string
                        fewshot_random_seed=1234,  # from default_seed_string
                        )

                    if results is not None:
                        new_results = {}
                        new_results["results"] = results['results']
                        new_results["groups"] = results["groups"]
                        new_results["group_subtasks"] = results["group_subtasks"]
                        model_string = model.split("/")[1]
                        with open(f"results/{model_string}_{revision}.json", 'w') as f:
                            json.dump(new_results, f)

                    print(
                        f"{model}, {revision} done"
                    )
    # # Save results
    # results_df.to_csv("model_evaluation_results.csv")
    # print("\nResults Summary:")
    # print(results_df)

if __name__ == "__main__":
    main()
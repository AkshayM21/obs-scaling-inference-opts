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
import os

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
        - se: Self Endorsement
        - mcts: Monte Carlo Tree Search
        """
    )
    parser.add_argument(
        '--sample',
        type=float,
        default=-1,
        help="""
        List percentage of entries to randomly sample from each benchmark for each optimization type.
        """
    )
    args = parser.parse_args()
    
    # Convert string 'None' to Python None
    if args.optimization == 'None':
        args.optimization = None
        
    return args

def main():
    args = parse_args()
    print(f"Selected optimization method: {args.optimization}")

    if args.optimization!=None and ',' in args.optimization:
        args.optimization = args.optimization.split(",")
    else:
        args.optimization = [args.optimization]
    
    models = [
        ("EleutherAI/pythia-160m-deduped", 1, 4),     
        #("EleutherAI/pythia-1.4b-deduped", 1, 4),   
        #("EleutherAI/pythia-410m-deduped", 1, 4),
        #("meta-llama/Llama-3.2-1B", 1, 4),
        #("EleutherAI/pythia-1b-deduped", 1, 4),
        #("allenai/OLMo-1B-0724-hf", ["main", "step5000-tokens10B", "step48000-tokens100B", "step477000-tokens1000B", "step954000-tokens2000B"], 1, 4),
        ("google/gemma-2-2b", 1, 4),
        ("meta-llama/Llama-3.2-3B", 1, 4),
        ("EleutherAI/pythia-2.8b-deduped", 1, 4),
        ("EleutherAI/pythia-6.9b-deduped", 2, 2),
        ("google/gemma-2-9b", 2, 2),
        ("Qwen/Qwen2.5-0.5B", 1, 4),
        ("Qwen/Qwen2.5-1.5B", 1, 4),
        ("Qwen/Qwen2.5-3B", 1, 4),
        ("Qwen/Qwen2.5-7B", 2, 2),
        ("allenai/OLMo-7B-0724-hf", ["main", "step2500-tokens10B", "step24000-tokens100B", "step239000-tokens1002B", "step477000-tokens2000B"], 2, 2),
        ("EleutherAI/pythia-12b-deduped", 4, 1),
        ("Qwen/Qwen2.5-14B", 4, 1),
        ("meta-llama/Llama-3.1-8B", 2, 2),
        ("google/gemma-2-27b", 4, 1),
        ("Qwen/Qwen2.5-32B", 4, 1), 
        ("meta-llama/Llama-3.1-70B", 4, 1),
        ("Qwen/Qwen2.5-72B", 4, 1),  
    ]


    beam_config = "num_beams=4,no_repeat_ngram_size=2,early_stopping=True,top_k=50,top_p=0.9,temperature=0.7,do_sample=True"

    tasks = ["mmlu", "hellaswag", "xwinograd", "winogrande", 
                "truthfulqa_mc1", "arc_challenge", "gsm8k"]


    gen_config = None
    task_manager = TaskManager("INFO", include_path=None)

    for model_tup in models:
        print(model_tup)
        if len(model_tup) == 3:
            model = model_tup[0]
            tp = model_tup[1]
            dp = model_tup[2]
            
            
            for opt in args.optimization:
                model_args = f"pretrained={model},"
                if opt == "beam":
                    gen_config = beam_config
                    model_args = model_args + "gen_config_provided=True," + beam_config
                elif opt == "sc":
                    pass
                elif opt == "cot":
                    model_args = model_args + "prompt_postpend=chain_of_thought"
                else:
                    model_args = model_args[:-1]
                
                
                print(
                f"starting {model}, {opt}"
                )
                results = simple_evaluate(model="hf-tpu", 
                    model_args=model_args,
                    tasks=tasks,
                    batch_size=16,
                    num_fewshot=None,
                    use_cache=None,
                    limit=None if args.sample == -1 else args.sample,
                    check_integrity=False,
                    write_out=False,
                    log_samples=(args.sample!=-1),
                    evaluation_tracker=None,
                    system_instruction=None,
                    apply_chat_template=False,
                    fewshot_as_multiturn=False,
                    gen_kwargs=gen_config,
                    task_manager=task_manager,
                    verbosity="INFO",
                    predict_only=False,
                    random_seed=0,
                    numpy_random_seed=1234,
                    torch_random_seed=1234,
                    fewshot_random_seed=1234,
                )

                if results is not None:
                    new_results = {}
                    new_results["results"] = results['results']
                    new_results["groups"] = results["groups"]
                    new_results["group_subtasks"] = results["group_subtasks"]
                    if args.sample != -1:
                        new_results["samples"] = results["samples"]
                    model_string = model.split("/")[1]
                    if args.sample != -1:
                        save_string = f"{opt}_results/{model_string}_limit{args.sample}.json"
                    else:
                        save_string = f"{opt}_results/{model_string}.json"
                    with open(save_string, 'w') as f:
                        json.dump(new_results, f)

                print(
                    f"{model} {opt} done"
                )
                
        else:
            model = model_tup[0]
            revisions = model_tup[1]
            tp = model_tup[2]
            dp = model_tup[3]
            
            for revision in revisions:
                
                
                for opt in args.optimization:
                    model_args = f"pretrained={model},revision={revision}"
                    if opt == "beam":
                        gen_config = beam_config
                        model_args = model_args + ",gen_config_provided=True,prompt_postpend=chain_of_thought," + beam_config
                    elif opt == "sc":
                        pass
                    elif opt == "cot":
                        model_args = model_args + ",prompt_postpend=chain_of_thought"
                        
                    print(
                    f"starting {model}, {revision}, {opt}"
                    )
                    results = simple_evaluate(model="hf-tpu", 
                        model_args=model_args,
                        tasks=tasks,
                        batch_size=16,
                        num_fewshot=None,
                        use_cache=None,
                        limit=None if args.sample == -1 else args.sample,
                        check_integrity=False,
                        write_out=False,
                        log_samples=(args.sample!=-1),
                        evaluation_tracker=None,
                        system_instruction=None,
                        apply_chat_template=False,
                        fewshot_as_multiturn=False,
                        gen_kwargs=gen_config,
                        task_manager=task_manager,
                        verbosity="INFO",
                        predict_only=False,
                        random_seed=0,
                        numpy_random_seed=1234,
                        torch_random_seed=1234,
                        fewshot_random_seed=1234,
                    )

                    if results is not None:
                        new_results = {}
                        new_results["results"] = results['results']
                        new_results["groups"] = results["groups"]
                        new_results["group_subtasks"] = results["group_subtasks"]
                        if args.sample != -1:
                            new_results["samples"] = results["samples"]
                        model_string = model.split("/")[1]
                        if args.sample != -1:
                            save_string = f"{opt}_results/{model_string}_{revision}_limit{args.sample}.json"
                        else:
                            save_string = f"{opt}_results/{model_string}_{revision}.json"
                        with open(save_string, 'w') as f:
                            json.dump(new_results, f)

                    print(
                        f"{model}, {revision}, {opt} done"
                    )

if __name__ == "__main__":
    main()

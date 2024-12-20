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
        - se: Self Endorsement
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
        
    
    # models = [
    #     ("EleutherAI/pythia-160m-deduped"),        
    #     ("EleutherAI/pythia-410m-deduped"),
    #     ("meta-llama/Llama-3.2-1B"),
    #     ("EleutherAI/pythia-1b-deduped"),
    #     ("allenai/OLMo-1B-0724-hf", ["main", "step5000-tokens10B", "step48000-tokens100B", "step477000-tokens1000B", "step954000-tokens2000B"]),
    #     ("google/gemma-2-2b"),
    #     ("EleutherAI/pythia-1.4b-deduped"),
    #     ("meta-llama/Llama-3.2-3B"),
    #     ("EleutherAI/pythia-2.8b-deduped"),
    #     ("EleutherAI/pythia-6.9b-deduped"),
    #     ("google/gemma-2-9b"),
    #     ("Qwen/Qwen2.5-0.5B"),
    #     ("Qwen/Qwen2.5-1.5B"),
    #     ("Qwen/Qwen2.5-3B"),
    #     ("Qwen/Qwen2.5-7B"),
    #     ("allenai/OLMo-7B-0724-hf", ["main", "step2500-tokens10B", "step24000-tokens100B", "step239000-tokens1002B", "step477000-tokens2000B"]),
    #     ("EleutherAI/pythia-12b-deduped"),
    #     ("Qwen/Qwen2.5-14B"),
    # ]
    models = [
        #("EleutherAI/pythia-160m-deduped", 1, 4),        
        # ("EleutherAI/pythia-410m-deduped", 1, 4),
        # ("meta-llama/Llama-3.2-1B", 1, 4),
        # ("EleutherAI/pythia-1b-deduped", 1, 4),
        # ("allenai/OLMo-1B-0724-hf", ["main", "step5000-tokens10B", "step48000-tokens100B", "step477000-tokens1000B", "step954000-tokens2000B"], 1, 4),
        ("google/gemma-2-2b", 1, 4),
        ("EleutherAI/pythia-1.4b-deduped", 1, 4),
        ("meta-llama/Llama-3.2-3B", 1, 4),
        ("EleutherAI/pythia-2.8b-deduped", 1, 4),
        ("EleutherAI/pythia-6.9b-deduped", 2, 2),
        ("google/gemma-2-9b", 2, 2),
        ("Qwen/Qwen2.5-0.5B", 1, 4),
        ("Qwen/Qwen2.5-1.5B", 1, 4),
        ("Qwen/Qwen2.5-3B", 1, 4),
        ("Qwen/Qwen2.5-7B", 2, 2),
        ("allenai/OLMo-7B-0724-hf", ["main", "step2500-tokens10B", "step24000-tokens100B", "step239000-tokens1002B", "step477000-tokens2000B"], 2, 2),
        ("EleutherAI/pythia-12b-deduped", 2, 2),
        ("Qwen/Qwen2.5-14B", 2, 2),
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
                model_args = "dtype=bfloat16,gpu_memory_utilization=0.8"
                if opt == "beam":
                    gen_config = beam_config
                    model_args = model_args + ",gen_config_provided=True,prompt_postpend=chain_of_thought," + beam_config
                elif opt == "sc":
                    pass
                elif opt == "cot":
                    model_args = model_args + ",prompt_postpend=chain_of_thought"
                
                print(
                f"starting {model}, {opt}"
                )
                results = simple_evaluate(model="vllm_reasoning", 
                    model_args=model_args+f",pretrained={model},tensor_parallel_size={tp},data_parallel_size={dp}",
                    tasks=tasks,
                    batch_size="auto",
                    num_fewshot=None,
                    use_cache=None,
                    limit=None,
                    check_integrity=False,
                    write_out=False,
                    log_samples=False,
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
                    model_string = model.split("/")[1]
                    with open(f"{opt}_results/{model_string}.json", 'w') as f:
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
                    model_args = "dtype=bfloat16,gpu_memory_utilization=0.8"
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
                    results = simple_evaluate(model="vllm_reasoning", 
                        model_args=model_args+f",pretrained={model},revision={revision},tensor_parallel_size={tp},data_parallel_size={dp}",
                        tasks=tasks,
                        batch_size="auto",
                        num_fewshot=None,
                        use_cache=None,
                        limit=None,
                        check_integrity=False,
                        write_out=False,
                        log_samples=False,
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
                        model_string = model.split("/")[1]
                        with open(f"{opt}_results/{model_string}_{revision}.json", 'w') as f:
                            json.dump(new_results, f)

                    print(
                        f"{model}, {revision}, {opt} done"
                    )

    # for opt in args.optimization:
    #     model_args = "dtype=bfloat16,gpu_memory_utilization=0.8"
    #     if opt=="beam":
    #         gen_config = beam_config
    #         model_args = model_args+",gen_config_provided=True,"+beam_config
    #         #task_manager = TaskManager("INFO", include_path="/home/ubuntu/obs-scaling-inference-opts/config/cot/")
    #         #tasks = ["mmlu", "hellaswag", "xwinograd", "winogrande", 
    #            # "truthfulqa_mc1", "arc_challenge", "gsm8k"] #CHANGE
    #         #tasks = ["hellaswag_generate", "arc_challenge_generate", "truthfulqa_generative", "xwinograd_generate", "winogrande_generate", "gsm8k", "mmlu_generative"] 
    #     elif opt=="sc":
    #         pass
    #         #self consistency with chain of thought
    #         #task_manager = TaskManager("INFO", include_path=None) #CHANGE
    #         #tasks = ["mmlu", "hellaswag", "xwinograd", "winogrande", 
    #         #    "truthfulqa_mc1", "arc_challenge", "gsm8k_cot_sc_new"] #CHANGE
    #     elif opt=="cot":
    #         model_args = model_args+",prompt_postpend=chain_of_thought"
    #         #chain of thought
    #         #task_manager = TaskManager("INFO", include_path="/home/ubuntu/obs-scaling-inference-opts/config/cot/")
    #         #tasks = ["mmlu", "hellaswag", "xwinograd", "winogrande", 
    #            # "truthfulqa_mc1", "arc_challenge", "gsm8k"] #CHANGE
    #         #tasks = ["hellaswag_cot", "arc_challenge_cot", "truthfulqa_cot", "xwinograd_cot", "winogrande_cot", "gsm8k_cot_zeroshot", "mmlu_flan_cot_zeroshot"] 
    #     for model_tup in models:
    #         print(model_tup)
    #         if len(model_tup)==3:
    #             model = model_tup[0]
    #             tp = model_tup[1]
    #             dp = model_tup[2]
    #             print(
    #                 f"starting {model}"
    #             )
    #             #no revisions
    #             results = simple_evaluate(model="vllm_reasoning", 
    #             model_args=model_args+f",pretrained={model},tensor_parallel_size={tp},data_parallel_size={dp}",
    #             tasks=tasks,
    #             batch_size="auto",
    #             num_fewshot=None,  # from default=None
    #             use_cache=None,  # from default=None
    #             limit=None,  # from default=None
    #             check_integrity=False,  # since it's an action='store_true' with no default
    #             write_out=False,  # from default=False
    #             log_samples=False,  # from default=False
    #             evaluation_tracker=None,  # unchanged as it's not in parser
    #             system_instruction=None,  # from default=None
    #             apply_chat_template=False,  # from default=False
    #             fewshot_as_multiturn=False,  # from default=False
    #             gen_kwargs=gen_config,  
    #             task_manager=task_manager,  # unchanged as it's not in parser
    #             verbosity="INFO",  # from default="INFO"
    #             predict_only=False,  # from default=False
    #             random_seed=0,  # from default_seed_string="0,1234,1234,1234"
    #             numpy_random_seed=1234,  # from default_seed_string
    #             torch_random_seed=1234,  # from default_seed_string
    #             fewshot_random_seed=1234,  # from default_seed_string
    #             )

    #             if results is not None:
    #                 new_results = {}
    #                 new_results["results"] = results['results']
    #                 new_results["groups"] = results["groups"]
    #                 new_results["group_subtasks"] = results["group_subtasks"]
    #                 model_string = model.split("/")[1]
    #                 with open(f"{opt}_results/{model_string}.json", 'w') as f:
    #                     json.dump(new_results, f)

    #             print(
    #                 f"{model} done"
    #             )
    #         else:
    #             model = model_tup[0]
    #             revisions = model_tup[1]
    #             tp= model_tup[2]
    #             dp = model_tup[3]
    #             for revision in revisions:
    #                 print(
    #                     f"starting {model}, {revision}"
    #                 )
    #                 results = simple_evaluate(model="vllm_reasoning", 
    #                     model_args=model_args+f",pretrained={model},revision={revision},tensor_parallel_size={tp},data_parallel_size={dp}",
    #                     tasks=tasks,
    #                     batch_size="auto",
    #                     num_fewshot=None,  # from default=None
    #                     use_cache=None,  # from default=None
    #                     limit=None,  # from default=None
    #                     check_integrity=False,  # since it's an action='store_true' with no default
    #                     write_out=False,  # from default=False
    #                     log_samples=False,  # from default=False
    #                     evaluation_tracker=None,  # unchanged as it's not in parser
    #                     system_instruction=None,  # from default=None
    #                     apply_chat_template=False,  # from default=False
    #                     fewshot_as_multiturn=False,  # from default=False
    #                     gen_kwargs=gen_config,
    #                     task_manager=task_manager,  # unchanged as it's not in parser
    #                     verbosity="INFO",  # from default="INFO"
    #                     predict_only=False,  # from default=False
    #                     random_seed=0,  # from default_seed_string="0,1234,1234,1234"
    #                     numpy_random_seed=1234,  # from default_seed_string
    #                     torch_random_seed=1234,  # from default_seed_string
    #                     fewshot_random_seed=1234,  # from default_seed_string
    #                     )

    #                 if results is not None:
    #                     new_results = {}
    #                     new_results["results"] = results['results']
    #                     new_results["groups"] = results["groups"]
    #                     new_results["group_subtasks"] = results["group_subtasks"]
    #                     model_string = model.split("/")[1]
    #                     with open(f"{opt}_results/{model_string}_{revision}.json", 'w') as f:
    #                         json.dump(new_results, f)

    #                 print(
    #                     f"{model}, {revision} done"
    #                 )
    # # # Save results
    # results_df.to_csv("model_evaluation_results.csv")
    # print("\nResults Summary:")
    # print(results_df)

if __name__ == "__main__":
    main()

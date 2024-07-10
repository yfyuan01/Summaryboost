# -*- coding: utf-8 -*-
#
# Run experiments on nine datasets collectied by TabLLM.
# Datasets:
#   ["bank", "blood", "calhousing", "car", "creditg", "diabetes", "heart", "income", "jungle"]
#
# Baselines:
#   TabLLM: prompting with k-shot in-context learning, wo/ finetuning
#   TabLet: generate a (prooetype) rule from examples as instruction 
#   SummaryBoost: generate summaries in an Adaboost pipeline
#
# cmds
#   python main.py --dataset all --llm llama3-8b --cv 0 --num_examples 128 --num_shots 16 --num_threads 8
#   python main.py --dataset all --llm mistral-7b --cv 0 --num_examples 128 --num_shots 16 --num_threads 8
#   python main.py --dataset all --llm gpt-3.5-turbo-0125 --cv 0 --num_examples 128 --num_shots 16 --num_threads 32
#

import argparse
import json
import os
import pandas as pd
import numpy as np


from llm_apis.common_api import parallel_invoke_llm, init_tracing
from llm_apis.eas_apis import init_eas_service
from helper.utils import load_base_instructions, load_rule_content, load_jsonl_with_keys
from helper.serilization import serialize_by_tabllm

from constants import TABLLM_DATASETS, INSTANCE_SEP, TABLLM_BINARY_DATASETS, MAX_CV
from selection_strategy import sv_select_max_abs, sv_select_balanced_max_abs,\
sv_select_systematic_sampling, sv_select_balanced_systematic_sampling


def tabllm(data_dir:str, dataset:str, cv:int, model:str, task_meta:dict,
           num_examples:int, num_shots:int, num_threads:int, optimization:str):
    """ 用LLM对dataset下的测试集进行推理，包括基础的tabllm方法以及我们在tabllm上增加了rule optimization之后的方法，
    请求完模型推理后，结果写到jsonl中，效果后续由evaluation.py统计。
    params:
        - cv (int): cross-validation id;
        - model (str): inference llm model, i.e., args.llm;
        - task_meta (json): background info of the dataset/task, c.f. task_instructions.json;
        - optimization: optimization to the base tabllm method, refer to our rule mining method
    
    return: inference results saved to a jsonl file.
    """
    ## init variables
    dataset_dir = os.path.join(data_dir, dataset, f"cv{cv}")
    description = task_meta["description"]
    question = task_meta["question"] 
    answer_choices = task_meta["answer_choices"]
    answer_requirement = task_meta["answer_requirement"]
    
    ## load train instance notes by tabllm
    train_path = os.path.join(dataset_dir, f"{dataset}_tabllm_cv{cv}train_{num_examples}.csv")
    train_examples = pd.read_csv(train_path)
    columns = list(train_examples.columns)

    # load the description (serialized data) of data samples
    notes = []
    for row in train_examples.itertuples(index=False):
        notes.append((serialize_by_tabllm(tuple(row), columns, question, answer_choices, is_train=True), row[-1]))

    # load testing examples 
    test_path = os.path.join(dataset_dir, f"{dataset}_tabllm_cv{cv}test_1000.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join(dataset_dir, f"{dataset}_tabllm_cv{cv}test.csv")
    test_examples = pd.read_csv(test_path)

    # if optimization is not None and optimization != ''
    if optimization:
        sink_suffix = f"ours{optimization}-{dataset}-cv{cv}-e{num_examples}-s{num_shots}"

        # if use svm-based optimization
        if optimization == "sv" or optimization == "cocktail":

            # load pre-calculated svm coefficients for data
            svcoef = load_jsonl_with_keys(
                dataset_dir, 
                filename=f"{dataset}-svcoef-cv{cv}-e{num_examples}",
                keys=None)

            # ==================================================================
            
            select = sv_select_systematic_sampling(svcoef, num_shots)
            if optimization == "cocktail":
                rule_content = load_rule_content(
                    dataset_dir,
                    f"{dataset}-ruleslr-cv{cv}-e{num_examples}-s{num_shots}")
            
            # ================================================================== 
            sv_notes = [notes[i] for i, _ in select]
            print(f"SVs: {len(sv_notes)} {[x[1] for x in sv_notes]}")

        elif optimization == "xgboost":
            entropy = load_jsonl_with_keys(
                dataset_dir, 
                filename=f"{dataset}-xgboostentropy-cv{cv}-e{num_examples}",
                keys=None)
            select = sv_select_max_abs(entropy, num_shots)
            rule_content = load_rule_content(
                dataset_dir,
                f"{dataset}-ruleslr-cv{cv}-e{num_examples}-s{num_shots}")
            xg_notes = [notes[i] for i, _ in select]
            print(f"XGBs: {len(xg_notes)} {[x[1] for x in xg_notes]}")

        # if use rule-based optimization
        elif optimization == "rule":
            rule_content = load_rule_content(dataset_dir, f"{dataset}-ruleslr-cv{cv}-e{num_examples}-s{num_shots}")
    else:
        sink_suffix = f"tabllm-{dataset}-cv{cv}-e{num_examples}-s{num_shots}"
    print(sink_suffix)

    num_proc, total_num_tokens = 0, 0
    prompts = []

    # iterate on test samples
    for row in test_examples.itertuples(index=False):
        current = serialize_by_tabllm(tuple(row), columns, question, answer_requirement, is_train=False)
        # system_prompt = "You are a helpful assistant good at table prediction.\n"
        if optimization == "sv":
            prompt = INSTANCE_SEP.join(
                [description] 
                + ['[FEW-SHOT EXAMPLES START]']
                + [_x[0] for _x in sv_notes]
                + ['[FEW-SHOT EXAMPLES END]']
                + ['[CURRENT QUESTION START]']
                + [current])

        elif optimization == "xgboost":
            patterns = f"Useful patterns for the task at hand:\n{rule_content}"
            prompt = INSTANCE_SEP.join(
                [patterns]
                + [description] 
                + ['[FEW-SHOT EXAMPLES START]']
                + [_x[0] for _x in xg_notes]
                + ['[FEW-SHOT EXAMPLES END]']
                + ['[CURRENT QUESTION START]']
                + [current])

        elif optimization == "cocktail":
            patterns = f"Useful patterns for the task at hand:\n{rule_content}"
            prompt = INSTANCE_SEP.join(
                [patterns]
                + [description]
                + ['[FEW-SHOT EXAMPLES START]']
                + [_x[0] for _x in sv_notes]
                + ['[FEW-SHOT EXAMPLES END]']
                + ['[CURRENT QUESTION START]']
                + [current])
        
        else:
            while True:
                np.random.shuffle(notes)
                if len(set([_x[1] for _x in notes[:num_shots]])) == len(answer_choices):
                    break
            prompt = INSTANCE_SEP.join(
                [description] 
                + ['[FEW-SHOT EXAMPLES START]']
                + [_x[0] for _x in notes[:num_shots]] 
                + ['[FEW-SHOT EXAMPLES END]']
                + ['[CURRENT QUESTION START]']
                + [current])

            if optimization == "rule":
                prompt = "Useful patterns for the task at hand:\n" + rule_content + INSTANCE_SEP + prompt

        # prompt = system_prompt + prompt
            
        if num_proc < 1:
            print(prompt)
            print(INSTANCE_SEP)
            
        total_num_tokens += len(prompt) // 4
        num_proc += 1
        prompts.append({
            "id": dataset + str(num_proc).zfill(5),
            "input": prompt,
            "label": row[-1],
            "answer": answer_choices[int(row[-1])]
        })
    
    print(f"# prompts: {len(prompts)}, # token: {total_num_tokens}, avg {total_num_tokens//num_proc}")
    parallel_invoke_llm(prompts, 
                        model=model, 
                        outfile=os.path.join(dataset_dir, f"{sink_suffix}-{model}"),
                        num_threads=num_threads, 
                        temperature=0.1, 
                        max_tokens=8,
                        num_invokes=0) 
    print(INSTANCE_SEP)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets_serialized", help="name of dataset to process")
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset to process")
    parser.add_argument("--llm", type=str, required=True, help="llm model")
    parser.add_argument("--cv", type=int, default=0, help="cross validation number. If cv = -1, then run for all cross validation datasets.")
    parser.add_argument("--num_examples", type=int, default=128, help="number of training examples")
    parser.add_argument("--num_shots", type=int, default=16, help="number of in-context learning shots")
    parser.add_argument("--num_threads", type=int, default=16, help="number of parallel threads")
    parser.add_argument("--optimization", type=str, default=None, help="optimization strategy: None/sv/rule/cocktail")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.llm.startswith("gpt"):
        # gpt-3.5-turbo-0125
        init_tracing()
    elif args.llm in ["mistral-7b", "llama3-8b"]:
        init_eas_service(data_dir="./llm_apis", filename="eas_services")
    instructions = load_base_instructions(filename="task_instructions.json") 
    datasets = [args.dataset] if args.dataset != "all" else TABLLM_BINARY_DATASETS  # TABLLM_DATASETS

    for dataset in datasets:
        if args.cv >= 0 and args.cv <= MAX_CV:
            tabllm(args.data_dir, dataset, cv=args.cv, model=args.llm, 
                   task_meta=instructions[dataset],
                   num_examples=args.num_examples, 
                   num_shots=args.num_shots, 
                   num_threads=args.num_threads,
                   optimization=args.optimization)
        elif args.cv == -1:
            for cv in range(MAX_CV+1):
                tabllm(args.data_dir, dataset, cv=cv, model=args.llm, 
                       task_meta=instructions[dataset],
                       num_examples=args.num_examples, 
                       num_shots=args.num_shots, 
                       num_threads=args.num_threads,
                       optimization=args.optimization)
        else:
            print(f"Invalid cv = {args.cv}")
            break

if __name__ == "__main__":
    main() 
# -*- coding: utf-8 -*-
#
# Compute performance metrics 
# 
# cmds
#   python evaluation.py --dataset all --cv 0

import argparse
import numpy as np
import os 
import tiktoken

from collections import defaultdict
from multiprocessing import Pool
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score


from utils import load_jsonl_with_keys, file_rank_key

gpt_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
notion_col_id = {"acc": 2, "error": 3, "bacc": 4, "prec": 5, "recall": 6, "f1": 7}
TABLLM_DATASETS = ["bank", "blood", "calhousing", "car", "creditg", "diabetes", "heart", "income", "jungle"]
# TABLLM_BINARY_DATASETS = ["bank", "blood", "calhousing", "creditg", "diabetes", "heart", "income", "jungle"]

def trans_pred_labels(records, dataset):
    num_err, y_true, y_pred = 0, [], []
    for idx, record in enumerate(records):
        err = False
        output = record["output"].strip()
        if dataset != "car":
            if output.startswith("No"):
                y_pred.append(0)
            elif output.startswith("Yes"):
                y_pred.append(1)
            else:
                # print(idx, record["output"])
                err = True
        else:
            if output.startswith("Unacceptable"):
                y_pred.append(0)
            elif output.startswith("Acceptable"):
                y_pred.append(1)
            elif output.startswith("Good"):
                y_pred.append(2)
            elif output.startswith("Very Good"):
                y_pred.append(3)
            else:
                err = True
        if err:
            num_err += 1
        else:
            y_true.append(int(record["label"]))

        
    return y_true, y_pred, len(records), num_err / len(records)


def token_len(input, _):
    return len(gpt_enc.encode(input))


def calc_gpt_tokens(inputs, num_proc=128):
    with Pool(processes=num_proc) as pool:
        result_objects = [
            pool.apply_async(
                func=token_len, 
                args=(_x, None)
            ) 
            for _x in inputs
        ]
        results = [r.get() for r in result_objects]

    return sum(results)


def my_metrics(y_true, y_pred, print_result=False):
    if print_result:
        print([(y_pred[i], y_true[i]) for i in range(len(y_true))])
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = (y_true == y_pred).sum() / (y_true.shape[0] + 1e-9)
    bacc = []
    prec, recall, f1 = {}, {}, {}
    for y in np.unique(y_true):
                
        prec[y] = np.sum((y_true == y) & (y_pred == y)) / (np.sum(y_pred == y) + 1e-9)
        recall[y] = np.sum((y_true == y) & (y_pred == y)) / (np.sum(y_true == y) + 1e-9)
        f1[y] = 2 * prec[y] * recall[y] / (prec[y] + recall[y] + 1e-9)
        bacc.append(recall[y])
    
    bacc = np.mean(bacc)
    if len(prec) == 2:
        prec, recall, f1 = prec[1], recall[1], f1[1]
    else:
        prec, recall, f1 = np.mean(list(prec.values())), np.mean(list(recall.values())), np.mean(list(f1.values()))

    return acc, bacc, prec, recall, f1


def main(args):
    datasets = TABLLM_DATASETS if args.dataset == "all" else [args.dataset]
    cvs = [x for x in range(5)] if args.cv == -1 else [args.cv]
    results = defaultdict(list)
    all_prompts = []
    print("File DSize ValidRatio Acc Error BAcc Prec Recall F1")
    for dataset in datasets:
        for cv in cvs:
            working_dir = os.path.join(args.data_dir, dataset, f"cv{cv}")
            files = [(file, file_rank_key(file)) for file in os.listdir(working_dir) 
                     if not file.startswith(dataset) and file.endswith(".jsonl") 
                     and (not args.llm or args.llm in file)]
            files.sort(key=lambda x: x[1])
            for file, _ in files:
                records = load_jsonl_with_keys(working_dir, file[:-6], keys=None, silent=True)
                all_prompts.extend([_x["input"] for _x in records])
                
                y_true, y_pred, dsize, err_ratio = trans_pred_labels(
                    records=records,
                    dataset=dataset
                )
                valid = 1. - err_ratio
                average_method = "binary" if dataset != "car" else "macro"
                labels = np.array([0, 1]) if dataset != "car" else np.array([0, 1, 2, 3])
                if y_true:
                    # acc = accuracy_score(y_true, y_pred)
                    # bacc = balanced_accuracy_score(y_true, y_pred)
                    # prec, recall, f1, _ = precision_recall_fscore_support(
                    #     y_true, y_pred, average=average_method, zero_division=0., labels=labels
                    # )
                    # print(" ".join([f"{x:.3f}" for x in my_metrics(y_true, y_pred, print_result=True)])) 
                    acc, bacc, prec, recall, f1 = my_metrics(y_true, y_pred, print_result=False)
                else:
                    acc, bacc, prec, recall, f1 = 0., 0., 0., 0., 0.
                
                # print(len(y_true), len(y_pred), err_ratio, prec, recall, f1)
                print(f"{file[:-6]} {dsize} {valid:.3f} {acc:.3f} {1.-acc:.3f} {bacc:.3f} {prec:.3f} {recall:.3f} {f1:.3f}")
                results[file[:-6].replace(f"-cv{cv}-", "-cvall-")].append(
                    (dsize, valid, acc, 1. - acc, bacc, prec, recall, f1)
                )
                # print()
            print()
    print()
    
    # print(f"Total gpt tokens: {calc_gpt_tokens(all_prompts)}\n\n")

    excel_results = []
    notion_results = defaultdict(dict)
    for key in results.keys():
        key_items = key.split('-')
        baseline, dataset = key_items[0], key_items[1]
        train_size, shots = key_items[3][1:], key_items[4][1:]
        llm = '-'.join(key_items[5:])
        dsize = int(np.mean([_x[0] for _x in results[key]]))
        valid, acc, error, bacc, prec, recall, f1 = [
            np.mean([_x[k] for _x in results[key]])
            for k in [1, 2, 3, 4, 5, 6, 7]
        ]
        excel_results.append((
            f"{str(shots).zfill(4)} {dataset} {str(train_size).zfill(4)}",
            f"{baseline} {dataset} {llm} {train_size} {shots} {dsize} {valid:.3f} {acc:.3f} {error:.3f} {bacc:.3f} {prec:.3f} {recall:.3f} {f1:.3f}"
        ))
        
        if train_size.isdigit():
            notion_results[f"{shots.zfill(4)} {dataset}"][int(train_size)] = [
                dsize, valid, acc, error, bacc, prec, recall, f1]

    excel_results.sort(key=lambda x: x[0])
    for _, details in excel_results:
        print(details) 
    print("\n")

    if args.notions:
        for col in args.notions:
            notion_prints = []
            for key, vals in notion_results.items():
                notion_vals, valids = [], []
                for train_size in sorted(vals.keys()):
                    notion_vals.append(vals[train_size][notion_col_id[col]])
                    valids.append(vals[train_size][1])
                
                notion_vals.append(np.mean(notion_vals))
                notion_vals.append(np.mean(valids))
                notion_prints.append([
                    key, 
                    ' '.join(f"{_x: .3f}" for _x in notion_vals)
                ])
            notion_prints.sort(key=lambda x: x[0])

            for key, content in notion_prints:
                print(col, key, content)
            print("\n")


def probe_data(args):
    error_outputs = set()
    datasets = TABLLM_DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        for num_examples in [32, 64, 128, 256]:
            for num_shots in [16, 32]:
                filename = f"tabllm-{dataset}-cv0-e{num_examples}-s{num_shots}-{args.llm}"
                records = load_jsonl_with_keys(os.path.join(args.data_dir, dataset, "cv0"), filename, keys=None)
                
                for record in records:
                    output = record["output"].lstrip()
                    if dataset != "car" and not output.startswith("Yes") and not output.startswith("No"):
                        # print(record["output"]) 
                        error_outputs.add(f"[{dataset}] {output}")
                    elif dataset == "car" and not output.startswith("Unacceptable") and not output.startswith("Acceptable")\
                        and not output.startswith("Good") and not output.startswith("Very Good"):
                        error_outputs.add(f"[{dataset}] {output}")

    error_outputs = sorted(list(error_outputs))
    for error_output in error_outputs:
        print(error_output)


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=str, default="./datasets_serialized", help="name of dataset to process")
        parser.add_argument("--dataset", type=str, required=True, help="name of dataset to process")
        parser.add_argument("--cv", type=int, default=0, help="cross validation number")
        parser.add_argument("--llm", type=str, default=None, help="only parse result by a specific llm")
        parser.add_argument("--notions", type=str, nargs='+', help="the result to display on notion")
        return parser.parse_args()

    # main(parse_args())
    probe_data(parse_args())
    
    
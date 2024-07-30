# -*- coding: utf-8 -*-
#
# Compute performance metrics 
# 
# cmds
#   python evaluation.py --dataset all --cv 0

import argparse
import numpy as np
import pandas as pd
import os 

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score

from constants import TABLLM_DATASETS, TABLLM_BINARY_DATASETS
from helper.utils import load_jsonl_with_keys, file_rank_key


def trans_pred_labels(records, dataset):
    num_err, y_true, y_pred, idx_list = 0, [], [], []
    for idx, record in enumerate(records):
        err = False
        output = record["output"].strip()
        idx_list.append(idx)
        if dataset != "car":
            if output.lower().startswith('yes') \
            or output.lower().endswith('yes.') \
            or output.lower().endswith("'yes'.") \
            or output.lower().find('answer: yes') >= 0 \
            or output.lower().find('answer is yes') >= 0 \
            or output.lower().find("answer is 'yes'") >= 0 \
            or output.lower().find("shows a heart disease") >= 0: # heart dataset
                y_pred.append(1)
            # elif output.lower().find('no') >= 0 or output.lower().find('avoid') >= 0:
            elif output.lower().startswith('no') \
            or output.lower().endswith('no.') \
            or output.lower().endswith("'no'.") \
            or output.lower().find('answer: no') >= 0 \
            or output.lower().find('answer is no') >= 0 \
            or output.lower().find("answer is 'no'") >= 0 \
            or output.lower().find("does not have coronary artery disease") >= 0:
                y_pred.append(0)
            else:
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
    return y_true, y_pred, len(records), num_err / max(len(records),1), idx_list




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets_serialized", help="name of dataset to process")
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset to process")
    parser.add_argument("--cv", type=int, default=0, help="cross validation number")
    parser.add_argument("--llm", type=str, default=None, help="only parse result by a specific llm")
    parser.add_argument("--method", type=str, default=None, help="cocktail/rule/sv. If empty, the method is tabllm (by default).")
    parser.add_argument("--num_shots", type=int, default=16, help="The number of learning shots. Default = 16")
    parser.add_argument("--num_examples", type=str, default='all', help="The number of learning examples. Default = 128")
    parser.add_argument("--additional_rules", type=str, default='True', help='whether evaluate for using additional rules. Default True')
    return parser.parse_args()

def add_eval_results(tb, file, dsize, valid, acc, error_rate, bacc, prec, recall, f1):
    tb['file'].append(file)
    tb['dsize'].append(dsize)
    tb['valid'].append(valid)
    tb['acc'].append(acc)
    tb['error_rate'].append(error_rate)
    tb['bacc'].append(bacc)
    tb['prec'].append(prec)
    tb['recall'].append(recall)
    tb['f1'].append(f1)

def main():
    args = parse_args()
    datasets = TABLLM_BINARY_DATASETS if args.dataset == "all" else [args.dataset]
    cvs = [x for x in range(5)] if args.cv == -1 else [args.cv]
    prefix = f"ours{args.method}" if args.method is not None else "tabllm"
    additional_rule_part = f"-additionalrule{args.additional_rules}-" if args.method is not None else ''
    tb_results = {
        "file":[],
        "dsize":[],
        "valid":[],
        "acc":[],
        "error_rate":[],
        "bacc":[],
        "prec":[],
        "recall":[],
        "f1":[]
    }

    print("File DSize ValidRatio ErrorRate BACC F1")
    for dataset in datasets:
        mean_results = {
            "file":[],
            "dsize":[],
            "valid":[],
            "acc":[],
            "error_rate":[],
            "bacc":[],
            "prec":[],
            "recall":[],
            "f1":[]
        }
        for cv in cvs:
            working_dir = os.path.join(args.data_dir, dataset, f"cv{cv}")
            files = [(file, file_rank_key(file)) for file in os.listdir(working_dir) 
                     if file.startswith(prefix)
                     and file.find(f"-s{args.num_shots}-") >= 0
                     and file.find(f"-e{args.num_examples}-") >= 0
                     and file.find(additional_rule_part) >= 0
                     and file.endswith(".jsonl") 
                     and (not args.llm or args.llm in file)]
            files.sort(key=lambda x: x[1])
            for file, _ in files:
                y_true, y_pred, dsize, err_ratio, _ = trans_pred_labels(
                    records=load_jsonl_with_keys(working_dir, file[:-6], keys=None, silent=True),
                    dataset=dataset
                )
                valid = 1. - err_ratio
                average_method = "binary" if dataset != "car" else "macro"
                if y_true:
                    acc = accuracy_score(y_true, y_pred)
                    error_rate = 1 - acc
                    bacc = balanced_accuracy_score(y_true, y_pred)
                    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average_method)
                    # print(" ".join([f"{x:.3f}" for x in my_metrics(y_true, y_pred, print_result=True)]))
                else:
                    acc, bacc, prec, recall, f1 = 0., 0., 0., 0., 0.
                    error_rate = 0.
                # print(len(y_true), len(y_pred), err_ratio, prec, recall, f1)
                # print(f"{file[:-6]} {dsize} {valid:.3f} {acc:.3f} {error_rate:.3f} {bacc:.3f} {prec:.3f} {recall:.3f} {f1:.3f}")
                # print(f"{file[:-6]}\t {dsize}\t {valid:.3f}\t {error_rate:.3f}\t {bacc:.3f}\t {f1:.3f}")
                add_eval_results(tb_results, file[:-6], dsize, valid, acc, error_rate, bacc, prec, recall, f1)
                add_eval_results(mean_results, file[:-6], dsize, valid, acc, error_rate, bacc, prec, recall, f1)     
        print(f"{mean_results['file'][0].replace('cv0','all')}\t {np.mean(mean_results['dsize']).round(1)}\t {np.mean(mean_results['valid']).round(3)}\t {np.mean(mean_results['error_rate']).round(3)}\t {np.mean(mean_results['bacc']).round(3)}\t {np.mean(mean_results['f1']).round(3)}")
            
    pd_results = pd.DataFrame(tb_results)
    pd_results.to_csv(f"evaluation_results/{prefix}-s{args.num_shots}-{args.llm}.csv")

if __name__ == "__main__":
    main()
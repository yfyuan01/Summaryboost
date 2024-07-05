# -*- coding: utf-8 -*-
# Author: renjun.hrj
# 2023-12-25

import json
import os
from openai import OpenAI

def preview_jsonl(filename, cnt):
    with open(f"{filename}.jsonl", 'r') as fin:
        num_preview = 0
        for line in fin:
            if not line.strip('\n'):
                continue
            d = json.loads(line.strip('\n'))
            print(d)
            print(d["input"])
            print("\n------\n")
            num_preview += 1
            if num_preview >= cnt:
                break 
    print(f"End of previewing {num_preview} examples.")


def load_jsonl_with_keys(data_dir, filename, keys, silent=False):
    data = []
    with open(os.path.join(data_dir, f"{filename}.jsonl"), 'r') as fin:
        for line in fin:
            if not line.strip('\n'):
                continue
            d = json.loads(line.strip('\n'))
            if not keys:
                data.append(d)
            else:
                data.append({key: d[key] for key in keys})
    
    if not silent:
        print(f"Load {len(data)} instances from {filename}.jsonl")
    return data

def sink_to_jsonl(data_dir, filename, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    with open(os.path.join(data_dir, f"{filename}.jsonl"), 'w') as fout:
        for record in data:
            json.dump(record, fout, ensure_ascii=False)
            fout.write('\n')
    print(f"Sink {len(data)} records into {filename}.jsonl")


def sink_to_json(data_dir, filename, data):
    with open(os.path.join(data_dir, f"{filename}.json"), 'w') as fout:
        json.dump(data, fout, indent=4)
    print(f"Sink data into {filename}.json")


def sink_to_csv(data_dir, filename, df, shuffle=False, keep=None):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    if keep is not None and isinstance(keep, int) and keep > 0:
        df = df.sample(frac=1).reset_index(drop=True).head(keep)
    df.to_csv(os.path.join(data_dir, f"{filename}.csv"), index=False)


def load_base_instructions(filename="task_instructions.json"):
    with open(filename, 'r') as file:
        inst = json.load(file)
    # print(inst)
    return inst 


def load_rule_content(data_dir, filename):
    data = load_jsonl_with_keys(data_dir, filename, keys=None)
    assert data[-1]["cmd"] == "summarization", "load_rules error."
    return data[-1]["output"]


def file_rank_key(filename):
    method, dataset, cv, num_examples, num_shots, llm = filename.split('-', 5)
    return f"{llm}-{method}-{dataset}-{cv}-{num_examples}-{num_shots}"
# -*- coding: utf-8 -*-
# 
# Tabular Prediction by LLM: accuracy and fairness
# 

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json
import numpy as np
import os
import time
import yaml

from helper.income import ZERO_SHOT, ZERO_SHOT_OPT
from helper.utils import preview_jsonl, load_jsonl_with_keys
from metrics import calc_metrics

ts = time.time()

def sink_raw_data_as_jsonl(dataset, sink_size):
    with open(os.path.join("./data", dataset, "config.yaml"), 'r') as file:
        config = yaml.safe_load(file)

    if config["type"] == "csv":
        json_data = []
        source =  config["test_file"]
        with open(os.path.join("./data", dataset, source), 'r') as file:
            for lid, line in enumerate(file):
                if config["header"] and lid == 0 or not line.strip('\n'):
                    continue 
                items = line.strip('\n').split(config["delimiter"])
                if len(items) != len(config["columns"]):
                    raise RuntimeError(f"Line format error in {line}")
                
                items = [_x.strip() if _x.strip() != '?' else "Unknown" for _x in items]
                line_json = {col["name"]: items[col["index"]] for col in config["columns"] if col["reserve"]}
                json_data.append(line_json)
    else:
        raise RuntimeError(f"Unknown data type {config['type']}")
    
    print(f"Load {len(json_data)} json lines from {source}")
    
    np.random.shuffle(json_data)
    num_sink = 0
    with open(os.path.join("./data", dataset, f"{source}.{sink_size}.jsonl"), 'w') as fout:
        for tmpj in json_data[:sink_size]:
            d = dict(tmpj)
            d["id"] = num_sink
            json.dump(d, fout, ensure_ascii=False)
            fout.write('\n')
            num_sink += 1
    
    print(f"Sink {num_sink} examples into {source}.{sink_size}.jsonl")

    return 


def serialize_to_prompt(dataset, source, output, prompt_templete):
    with open(os.path.join("./data", dataset, f"{source}.jsonl"), 'r') as fin,\
        open(os.path.join("./data", dataset, f"{output}.jsonl"), 'w') as fout:
        for line in fin:
            if not line.strip('\n'):
                continue 
            d = json.loads(line.strip('\n'))
            prompt = prompt_templete.format(**d)
            d["input"] = prompt
            json.dump(d, fout, ensure_ascii=False)
            fout.write('\n')
    print(f"serialize_to_prompt from {source} to {output} complete.")
    return


def request_gpt(dataset, source, model, num_threads=1):
    prompts = load_jsonl_with_keys(data_dir=os.path.join("./data", dataset), filename=source, keys=None)

    invoke_gpt_with_defaults = partial(
        invoke_gpt,
        model=model,
        system_prompt="You are a helpful assistant.",
        temperature=0.,
        max_tokens=2048,
    )
    num_completed = 0
    if model == "gpt-3.5-turbo-1106":
        out_filename = f"{source}.gpt35_1106"
    elif model == "gpt-3.5-turbo-0613":
        out_filename = f"{source}.gpt35_0613"
    elif model in ["gpt-4-1106-preview"]:
        out_filename = f"{source}.gpt4"
    else:
        raise ValueError(f"Unknown GPT model {model}")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor,\
        open(os.path.join("./data", dataset, f"{out_filename}.jsonl"), 'w') as fout:
        futures = [executor.submit(invoke_gpt_with_defaults, prompt) for prompt in prompts]
        
        for future in as_completed(futures):
            result = future.result()
            json.dump(result, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()

            num_completed += 1
            if num_completed % 100 == 0:
                print(f"Complete {num_completed}/{len(prompts)} Tasks in {time.time() - ts:.1f} secs.", flush=True)
    
        print(f"Complete {num_completed}/{len(prompts)} Tasks in {time.time() - ts:.1f} secs.")


def result_analysis(dataset, source):
    examples = load_jsonl_with_keys(data_dir=os.path.join("./data", dataset), filename=source, keys=None)
    print(f"Dataset: {dataset} {source}")
    acc, precision, recall, f1 = calc_metrics(dataset="income", examples=examples)
    print(
        "Gender=All\n"
        f"Accuracy: {acc: .3f}\n"
        f"F1: {f1:.3f}, precision: {precision:.3f} / recall: {recall:.3f}\n"
    )

    for gender in set(x['sex'] for x in examples):
        acc, precision, recall, f1 = calc_metrics(dataset="income", examples=examples, gender=gender)
        print(
            f"Gender={gender}\n"
            f"Accuracy: {acc: .3f}\n"
            f"F1: {f1:.3f}, precision: {precision:.3f} / recall: {recall:.3f}\n"
        )

    print("End of results.\n")

def main():
    # sink_raw_data_as_jsonl(dataset="income", sink_size=2000)
    # serialize_to_prompt(dataset="income", source="adult.test.2000", output="adult.test.2000.zs", prompt_templete=ZERO_SHOT)
    # serialize_to_prompt(dataset="income", source="adult.test.2000", output="adult.test.2000.zso", prompt_templete=ZERO_SHOT_OPT)

    # preview_jsonl(os.path.join("./data", "income", "adult.test.2000.zs"), cnt=5)
    # preview_jsonl(os.path.join("./data", "income", "adult.test.2000.zso"), cnt=5)

    # # model="gpt-3.5-turbo-1106"
    # model = "gpt-3.5-turbo-0613"
    # request_gpt(dataset="income", source="adult.test.2000.zs", model=model, num_threads=20)
    # request_gpt(dataset="income", source="adult.test.2000.zso", model=model, num_threads=20)

    # result_analysis(dataset="income", source="adult.test.2000.zs.gpt35_0613")
    # result_analysis(dataset="income", source="adult.test.2000.zso.gpt35_0613")

    # result_analysis(dataset="income", source="adult.test.2000.zs.gpt35_1106")
    # result_analysis(dataset="income", source="adult.test.2000.zso.gpt35_1106")

    result_analysis(dataset="income", source="adult.test.2000.zsoc.gpt35_0613")
    result_analysis(dataset="income", source="adult.test.2000.zsoc.v2.gpt35_0613")
    
    result_analysis(dataset="income", source="adult.test.2000.zsoe.gpt35_0613")
    result_analysis(dataset="income", source="adult.test.2000.zsoe.v2.gpt35_0613")

if __name__ == "__main__":
    main()
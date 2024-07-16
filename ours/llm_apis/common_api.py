# -*- coding: utf-8 -*-
#
# Common LLM APIs for OpenAI, PAI EAS, ...
# 

import json
import os
import numpy as np
import time 


from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from llm_apis.openai_apis import invoke_gpt
from llm_apis.eas_apis import invoke_eas


def init_tracing():
    # from llm_trace.instrument.pop_exporter import PopExporter
    # from llm_trace.instrument.instrumentor import OpenAIInstrumentor

    # OpenAIInstrumentor(
    #     PopExporter(
    #         llm_app_name='llm-trace-for-tabular',
    #         llm_app_version='0.0.1',
    #         ak=os.environ.get("LLM_TRACING_AK"),
    #         sk=os.environ.get("LLM_TRACING_SK"),
    #         region='cn-hangzhou',
    #         endpoint='paillmtrace-pre.cn-hangzhou.aliyuncs.com',
    #     )
    # ).instrument()
    pass


def determine_invoke_func(model, temperature, max_tokens):
    if model.startswith("gpt"):
        return partial(
            invoke_gpt,
            model=model,
            system_prompt="You are a helpful assistant.",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model in ["mistral-7b", "llama3-8b"]:
        return partial(
            invoke_eas,
            model=model,
            system_prompt="You are a helpful assistant.",
            input_col="input",
            output_col="output",
            temperature=temperature,
            max_new_tokens=max_tokens,
        ) 
    else:
        raise RuntimeError(f"Unknown service not available for llm model {model}")


def parallel_invoke_llm(prompts, model, outfile, num_threads=1, temperature=0.2, max_tokens=4096, num_invokes=0):
    """ parallel invoke llm and sink results into a disk file.
    """
    if num_invokes > 0:
        np.random.shuffle(prompts)
        prompts = prompts[:num_invokes]
        print(f"Remaining {len(prompts)} API calls after random sampling")

    invoke_func_with_defaults = determine_invoke_func(model, temperature, max_tokens)
    num_completed = 0
    ts = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor,\
        open(f"{outfile}.jsonl", 'w') as fout:
        futures = [executor.submit(invoke_func_with_defaults, prompt) for prompt in prompts]
        
        for future in as_completed(futures):
            result = future.result()
            json.dump(result, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()

            num_completed += 1
            if num_completed % 100 == 0:
                print(f"Complete {num_completed}/{len(prompts)} prompts in {time.time() - ts:.1f} secs.", flush=True)
    
        print(f"Complete {num_completed}/{len(prompts)} prompts in {time.time() - ts:.1f} secs.")


def batch_invoke_llm(prompts, model, num_threads=1, temperature=0.2, max_tokens=4096):
    """ parallel invoke llm and directly return results.
    """

    invoke_func_with_defaults = determine_invoke_func(model, temperature, max_tokens)
    ts = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(invoke_func_with_defaults, prompt) for prompt in prompts]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
        print(f"Complete {len(prompts)} prompts in {time.time() - ts:.1f} secs.")
    
    return results
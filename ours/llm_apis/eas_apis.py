# -*- coding: utf-8 -*-
#
# Aliyyn PAI EAS APIs
# 
# python webui/webui_server.py \
#   --port=8000 \
#   --model-path=meta-llama/Meta-Llama-3-70B-Instruct \
#   --backend=vllm \
#   --gpu-memory-utilization 0.95 \
#   --max-model-len 8192

import os
import json
import requests
from helper.utils import load_jsonl_with_keys


def init_eas_service(data_dir, filename):
    services = load_jsonl_with_keys(data_dir=data_dir, filename=filename, keys=None)
    models = []
    for service in services:
        model, endpoint, token, seqlen = service["model"], service["endpoint"], service["token"], service["seqlen"]
        if not "offline" in model:
            os.environ[f"{model}_ENDPOINT"] = endpoint 
            os.environ[f"{model}_TOKEN"] = token 
            os.environ[f"{model}_SEQLEN"] = str(seqlen)
            models.append(model)
    print(f"Init {len(models)} eas services: {models}")


def invoke_eas(prompt: dict,
               model: str,
               system_prompt: str,
               input_col: str,
               output_col: str,
               max_new_tokens: int = 4096,
               temperature: float = 0.2,
               top_k: int = 100,
               top_p: float = 1.,
               frequency_penalty: float = 0.,
               presence_penalty: float = 0.,
               lang: str = "English") -> dict:
    """ Invoking LLM EAS service for single-turn chat.
    The query to LLM is specified in prompt[input_col], and the response should be written to 
    prompt[output_col]. Finally, the appended prompt dict is returned.
    """
    try:
        endpoint = os.environ.get(f"{model}_ENDPOINT")
        token = os.environ.get(f"{model}_TOKEN")
        seqlen = int(os.environ.get(f"{model}_SEQLEN"))
    except:
        raise RuntimeError("EAS service of {model} not found")
    
    headers = {
        "Authorization": token,
    }


    est_input_tokens = len(prompt[input_col]) if lang == "Chinese" else len(prompt[input_col]) // 4

    pload = {
        "prompt": prompt[input_col],
        "system_prompt": prompt["sys_input"] if "sys_input" in prompt else system_prompt,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "max_new_tokens": max(min(max_new_tokens, seqlen - 100 - est_input_tokens), 128),
        "use_stream_chat": False,
        "history": [],
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "do_sample": True,
        "use_cache": True
    }
    prompt["llm_model"] = model

    try:
        response = requests.post(endpoint, headers=headers, json=pload, stream=False, timeout=120)
        data = json.loads(response.content)
        prompt[output_col] = data["response"]
    except (TimeoutError, requests.exceptions.ReadTimeout):
        id = prompt["id"] if "id" in prompt else ""
        print(f"EAS Timeout Error during processing {id}")
        prompt[output_col] = "Error: 请求超时，服务不可用。Request Timeout. Service unavailable."
    except:
        id = prompt["id"] if "id" in prompt else ""
        print(f"EAS Runtine Error during processing {id}")
        prompt[output_col] = "Error: 服务运行失败。Runtime Error."

    return prompt
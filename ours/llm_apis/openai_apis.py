# -*- coding: utf-8 -*-
#
# OpenAPIs
# 

from openai import OpenAI


def invoke_gpt(prompt, model, system_prompt="You are a helpful assistant.", temperature=0.8, max_tokens=4096):
    client = OpenAI()
    prompt["llm_model"] = model

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt["sys_input"] if "sys_input" in prompt else system_prompt},
                {"role": "user", "content": prompt["input"]}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            # frequency_penalty=0,
            # presence_penalty=0,
        )
        prompt["output"] = completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e} on {prompt['input']}")
        prompt["output"] = "Error: " + str(e)

    return prompt

#this is the script for the original baseline
import pandas as pd
import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
import random
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from eas_apis import init_eas_service, invoke_eas
import pickle
import numpy as np
client = OpenAI()
model = 'gpt-3.5-turbo'
# model = 'llama3-8b'

def my_metrics(y_true, y_pred, print_result=False):
# This is borrowed from renjun
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
        prec, recall, f1 = prec['True'], recall['True'], f1['True']
    else:
        prec, recall, f1 = np.mean(list(prec.values())), np.mean(list(recall.values())), np.mean(list(f1.values()))

    return acc, bacc, prec, recall, f1


def read_meta(meta_info):
    description = f"Title: {meta_info['title']}."
    y_dict = {meta_info['answer_choices'][1]:True,meta_info['answer_choices'][0]:False}
    summaryprompt = meta_info['summarization']
    ending = meta_info['answer_requirement']
    question = meta_info['question']
    return description,y_dict,summaryprompt,ending,question
def Summary(X,y,shot=4):
    r = random.sample(range(len(X)), shot)
    X_s = [X[i] for i in r]
    y_s = [y[i] for i in r]
    return 'Example: '.join([X_s[i] + '\nPrediction: '+str(y_s[i])+'\n' for i in range(len(X_s))])

def Original(X_train,y_train, X_test, y_test, shot, metadata,description,ending,question):
    sample = Summary(X_train, y_train,shot)
    error_rate, f1_rate, error_list = error_cal(sample, X_test, y_test, description, metadata,  ending, question)
    print(f'final error rate: {error_rate}')
    print(f'final f1 rate: {f1_rate}')
    return error_rate, error_list
def DataPreprocessor(file_address):
    df = pd.read_csv(file_address)
    ndf = df.to_numpy()
    col = df.columns
    y = df[col[-1]]
    text = []
    for r in ndf:
        text.append('\t'.join([col[i]+': '+str(r[i]) for i in range(len(r)-1)])) #add threshold
    return text,y

def error_cal(summary, X, y, description, metadata, ending, question):
    error = 0
    elist = []
    a_list = []
    prompts = []
    results = []
    for k,query in tqdm(enumerate(X)):
        prompt = f"{description}\n{metadata}\nExample: {summary}\nNow here's the question: {query}\n{question}\n{ending}"
        prompts.append(prompt)
        if model == 'gpt-3.5-turbo':
            s = generate(prompt)
            results.append(s)
    if model != 'gpt-3.5-turbo':
        results = generate_osllm_batch(prompts)
    for s in results:
        # try:
        #     answer = s
        # except:
        if len(set(y))==2:
            if s.lower().find('yes')>=0 or s.lower().find('true')>=0 :
                answer = True
            elif s.lower().find('no')>=0 or s.lower().find('false')>=0:
                answer = False
            else:
                print(s)
                answer='Not sure'
        else:
            if s.lower().find('unacceptable')>=0:
                answer = 0
            elif s.lower().find('acceptable')>=0:
                answer = 1
            elif s.lower().find('good')>=0 and s.lower().find('very good')<0:
                answer = 2
            elif s.lower().find('very good')>=0:
                answer = 3
            else:
                print(s)
                answer = -1
        a_list.append(answer)
        if str(answer)!=str(y[k]):
            error += 1
            elist.append(1)
        else:
            elist.append(0)

    a_list = [str(a) for a in a_list]
    y = [str(p) for p in y]
    acc, bacc, prec, recall, f1 = my_metrics(y, a_list)
    return error/float(len(X)), f1, elist

def generate(prompt):
    conversation = [{"role": "system", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=conversation
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
def generate_osllm_batch(prompts, num_threads=16, max_tokens=4096, temperature=0.2, num_invokes=0):
    """ parallel invoke llm and sink results into a disk file {outfile}.jsonl

    Args:
        prompts: a list of dict, each dict d is a request, d["input"] is the prompt
        model: llama3-8b / mistral-7b
        outfile: results will be saved to {outfile}.jsonl
        num_threads: concurrent threads for speedup, set to 16
    """
    if not prompts:
        return
    if isinstance(prompts[0], str):
        prompts = [{"input": prompt} for prompt in prompts]
    if num_invokes > 0:
        random.shuffle(prompts)
        prompts = prompts[:num_invokes]
        print(f"Remaining {len(prompts)} API calls after random sampling")

    invoke_func_with_defaults = partial(
        invoke_eas,
        model=model,
        system_prompt="You are a helpful assistant.",
        input_col="input",
        output_col="output",
        temperature=temperature,
        max_new_tokens=max_tokens,
    )
    print(model)
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(invoke_func_with_defaults, prompt) for prompt in prompts]
        for future in as_completed(futures):
            result = future.result()
            results.append(result["output"])
    return results

if __name__ == '__main__':
    file_folder = '../datasets_serialized'
    meta_file_address = 'task_instructions.json'
    meta_info = json.load(open(meta_file_address))
    description_data = open('task_description.txt').readlines()
    description_data = {i.split(':')[0]: i.split(':')[1].strip('\n').lstrip(' ') for i in description_data}
    parser = argparse.ArgumentParser(description='Some arguments')
    parser.add_argument('--portion', type=str, default='32', choices=['16','32', '64', '128', '256', 'all'])
    parser.add_argument('--category', type=str, default='diabetes',
                        choices=['bank', 'blood', 'calhousing', 'car', 'creditg', 'diabetes', 'heart', 'income',
                                 'jungle'])
    parser.add_argument('--shot', type=int, default=16)
    args = parser.parse_args()
    portion = args.portion
    shot = args.shot
    for category in [args.category]:
        for j in range(5):
            print(j)
            metadata = description_data[category]
            description, y_dict, summaryprompt, ending, question = read_meta(meta_info[category])
            if category not in ['income', 'jungle', 'calhousing', 'bank']:
                test_file = os.path.join(file_folder,
                                           f'{category}/cv{str(j)}/{category}_tabllm_cv{str(j)}test.csv')
            else:
                test_file = os.path.join(file_folder,
                                           f'{category}/cv{str(j)}/{category}_tabllm_cv{str(j)}test_1000.csv')

            X_test,y_test = DataPreprocessor(test_file)
            X_train,y_train = DataPreprocessor(os.path.join(file_folder,category,f'cv{str(j)}',f'{category}_tabllm_cv{str(j)}train_{portion}.csv'))
            error_rate, error_list = Original(X_train, y_train, X_test, y_test, shot, metadata, description, ending, question)
            output_address = os.path.join(file_folder,
                                          f'{category}/cv{str(j)}/original_{category}_cv{str(j)}test_{portion}_{model}.pkl')
            with open(output_address, 'wb') as f:
                pickle.dump({'error_rate': error_rate, 'error_list': error_list}, f, protocol=1)

        # answers = Preprocessor(X,y,meta_data[category],y_dict)

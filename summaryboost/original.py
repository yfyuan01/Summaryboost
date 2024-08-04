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
from eas_apis import init_eas_service,invoke_eas
# from ours.llm_apis.eas_apis import init_eas_service
import pickle
import numpy as np
from copy import deepcopy
client = OpenAI()
init_eas_service(data_dir="./ours/llm_apis", filename='eas_services')


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
    f1_dict = deepcopy(f1)
    bacc = np.mean(bacc)
    if len(prec) == 2:
        prec, recall, f1 = prec['True'], recall['True'], f1['True']
    else:
        prec, recall, f1 = np.mean(list(prec.values())), np.mean(list(recall.values())), np.mean(list(f1.values()))

    return acc, bacc, prec, recall, f1, f1_dict


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
    return 'Example: '.join([X_s[i] + '\nAnswer: '+str(y_s[i])+'\n' for i in range(len(X_s))])

def Original(X_train,y_train, X_test, y_test, shot, metadata,description,ending,question,model,y_dict):
    sample = Summary(X_train, y_train,shot)
    error_rate, f1_rate, error_list, f1_dict, results = error_cal(sample, X_test, y_test, description, metadata,  ending, question, model,y_dict)
    print(f'final error rate: {error_rate}')
    print(f'final f1 rate: {f1_rate}')
    print(f'final f1 list: {f1_dict}')
    return error_rate, error_list, f1_rate, f1_dict, results
def DataPreprocessor(file_address):
    if file_address.endswith('.csv'):
        df = pd.read_csv(file_address)
        ndf = df.to_numpy()
        col = df.columns
        y = df[col[-1]]
        text = []
        for r in ndf:
            text.append('\t'.join([col[i]+': '+str(r[i]) for i in range(len(r)-1)])) #add threshold
        return text,y
    else:
        examples = [json.loads(line.strip('\n')) for line in open(file_address).readlines()]
        rows = [e['row']for e in examples]
        cols = [e['column'] for e in examples]
        y = [r[-1] for r in rows]
        text = []
        for col,r in zip(cols,rows):
            text.append('\t'.join([col[i] + ': ' + str(r[i]) for i in range(len(r) - 1)]))  # add threshold
        return text, y

def error_cal(summary, X, y, description, metadata, ending, question, model,y_dict):
    error = 0
    elist = []
    a_list = []
    prompts = []
    results = []
    candidates = '/'.join(list(y_dict.keys()))
    for k,query in tqdm(enumerate(X)):
        prompt = f"{description}\n{metadata}\n###\n[FEW-SHOT EXAMPLES START]\n{summary}\n[FEW-SHOT EXAMPLES END]\n###\n[CURRENT QUESTION START]\nNow here's the question: {query}\n{question}\n{ending}\nAnswer: <xxx, {candidates}>"
        prompts.append(prompt)
        if model == 'gpt-3.5-turbo':
            s = generate(prompt)
            results.append(s)
    if model != 'gpt-3.5-turbo':
        results = generate_osllm_batch(prompts,model,num_threads=16)
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
                # print(s)
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
                # print(s)
                answer = -1
        a_list.append(answer)
        if str(answer)!=str(y[k]):
            error += 1
            elist.append(1)
        else:
            elist.append(0)

    a_list = [str(a) for a in a_list]
    y = [str(p) for p in y]
    acc, bacc, prec, recall, f1, f1_dict = my_metrics(y, a_list)
    return error/float(len(X)), f1, elist, f1_dict, results

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
def generate_osllm_batch(prompts, model, num_threads=16, max_tokens=4096, temperature=0.2, num_invokes=0):
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
    parser.add_argument('--category', type=str, default='bank',
                        choices=['bank', 'blood', 'calhousing', 'car', 'creditg', 'diabetes', 'heart', 'income',
                                 'jungle', 'all'])
    parser.add_argument('--model', type=str, default='mistral-7b',)
    parser.add_argument('--shot', type=int, default=16)
    parser.add_argument("--shuffled", type=str, default='False', help='whether evaluate for shuffled version of testing set. Default False')
    parser.add_argument("--prob", type=float, default=None, help="number of true sample portion")
    args = parser.parse_args()
    portion = args.portion
    shot = args.shot
    model = args.model
    # model = 'gpt-3.5-turbo'
    # model = 'mistral-7b'
    # model = 'llama3-8b'
    print(f'Using the model: {model}')
    print(f'Using the portion: {portion}')
    if args.category == 'all':
        all_list = ['bank', 'blood', 'calhousing', 'car', 'creditg', 'diabetes', 'heart', 'income','jungle']
    else:
        all_list = [args.category]
    if args.prob != None:
        all_list = ['bank']
    for category in all_list:
    # for category in ['bank', 'blood', 'calhousing', 'car', 'creditg', 'diabetes', 'heart', 'income']:
        print(f'category: {category}')
        f_score_true = 0.
        f_score_false = 0.
        for j in range(5):
            print(j)
            metadata = description_data[category]
            description, y_dict, summaryprompt, ending, question = read_meta(meta_info[category])
            if args.prob != None:
                test_file = os.path.join(file_folder,
                                        f'{category}/cv{str(j)}/{category}_tabllm_cv{str(j)}test_1000_p{args.prob}.csv')
            if args.shuffled=='True':
                if category not in ['income', 'jungle', 'calhousing', 'bank']:
                    test_file = os.path.join(file_folder,
                                             f'{category}/cv{str(j)}/{category}_tabllm_cv{str(j)}test_shuffled.jsonl')
                else:
                    test_file = os.path.join(file_folder,
                                             f'{category}/cv{str(j)}/{category}_tabllm_cv{str(j)}test_1000_shuffled.jsonl')
            elif args.prob == None and args.shuffled=='False':
                if category not in ['income', 'jungle', 'calhousing', 'bank']:
                    test_file = os.path.join(file_folder,
                                             f'{category}/cv{str(j)}/{category}_tabllm_cv{str(j)}test.csv')
                else:
                    test_file = os.path.join(file_folder,
                                             f'{category}/cv{str(j)}/{category}_tabllm_cv{str(j)}test_1000.csv')
            print(test_file)
            X_test,y_test = DataPreprocessor(test_file)
            X_train,y_train = DataPreprocessor(os.path.join(file_folder,category,f'cv{str(j)}',f'{category}_tabllm_cv{str(j)}train_{portion}.csv'))
            error_rate, error_list, f1, f1_dict, results = Original(X_train, y_train, X_test, y_test, shot, metadata, description, ending, question, model,y_dict)
            f_score_true+=f1
            f_score_false+=f1_dict['False']
            if args.shuffled=='False' and args.prob == None:
                output_address = os.path.join(file_folder,
                                              f'{category}/cv{str(j)}/original_{category}_cv{str(j)}test_{portion}_{model}_{shot}.pkl')
            elif args.prob!= None:
                output_address = os.path.join(file_folder,
                                              f'{category}/cv{str(j)}/original_{category}_cv{str(j)}test_{portion}_{model}_{shot}_p{args.prob}.pkl')
            else:
                output_address = os.path.join(file_folder,
                                          f'{category}/cv{str(j)}/original_{category}_cv{str(j)}test_{portion}_{model}_{shot}_shuffled.pkl')
            with open(output_address, 'wb') as f:
                pickle.dump({'error_rate': error_rate, 'error_list': error_list, 'f1_dict':f1, 'results':results}, f, protocol=1)
        print(f'average f1 true:{f_score_true/5.0}')
        print(f'average f1 false:{f_score_false / 5.0}')
        # answers = Preprocessor(X,y,meta_data[category],y_dict)

import csv
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from openai import OpenAI
import random
from collections import Counter
import os
import pickle
from tqdm import tqdm
import json
import csv
import random
import time 
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from eas_apis import init_eas_service, invoke_eas

client = OpenAI()
init_eas_service()

# metadata = """Description: Data taken from the Blood Transfusion Service Center in Hsin-Chu City in Taiwan - this is a classification problem. The goal is to predict whether a given individual will consent or avoid donating blood. It includes the regency - months since last donation, frequency - total number of donation, Monetary - total blood donated in c.c., and Time - months since first donation.\n"""
# summaryprompt = """Tl;dr """
# description = """Title: Blood donation Prediction.\n"""
# ending = """Therefore, this individual is likely to (avoid/consent): """
model_version = 'gpt-3.5-turbo'
# data_file_address = '/Users/yuanyifei/Downloads/blood/blood_raw.csv'
# y_dict = {'consent':True, 'avoid':False}
def read_meta(meta_info):
    description = f"Title: {meta_info['title']}."
    y_dict = {meta_info['answer_choices'][1]:True,meta_info['answer_choices'][0]:False}
    summaryprompt = meta_info['summarization']
    ending = meta_info['answer_requirement']
    question = meta_info['question']
    return description,y_dict,summaryprompt,ending,question
def ClusterSampling(X,y,r,p,s,portion,run,dataset):
    S, Y = [], []
    w = [-1,]*len(X)
    for k in range(len(set(y))):
        E = GPTEmbedding([X[l] for l in range(len(y)) if y[l]==list(set(y))[k]],k,portion,run,dataset) #mark
        idx = [l for l in range(len(X)) if y[l]==list(set(y))[k]]
        idx_d = {i:idx[i] for i in range(len(idx))}
        C = AgglomerativeClustering().fit(E)
        labels = C.labels_.tolist()
        cl = Counter(labels)
        c = [None]*len(labels)
        for j in range(len(labels)):
            c[j] = len(X)/float(cl[labels[j]])
        # for i in range(len(labels)):
            w[idx_d[j]] = c[j]
        # for i in range(len(X)):
        #     w[i] = c[i] #mark
        w = normalize(normalize([w])*p)[0].tolist()
        #Sample s*r[c]
        x_s, y_s = sample(X,y,int(s*r[k]),w)
        S.extend(x_s)
        Y.extend(y_s)
    return S, Y

def sample(X,y,n,w):
    idx = random.choices(range(len(X)), weights=w, k=n)
    return [X[i] for i in idx], [y[i] for i in idx]

def Summaryboost(X_train,y_train, X_test, y_test, T,s,k,metadata,summaryprompt,description,y_dict,ending,question,portion,run,dataset, model):
    """
    X: all training data
    y: all training label
    T: maximum number of rounds
    s: size of the sampling subset
    r: ratio of classes
    """
    h, epsilon, alpha = [None]*T, [None]*T, [None]*T
    N = len(X_train)
    K = len(set(y_train))
    w = [1.0/N,]*N
    for r in range(0, T):
        while epsilon[r]==None or epsilon[r]>=(1-1.0/K):
            Xs, Ys = ClusterSampling(X_train,y_train,k,w,s,portion,run,dataset)
            h[r] = Summary(Xs,Ys,X_train,y_train, metadata,summaryprompt,description,y_dict,ending,question,model)
            error_rate, error_list = error_cal(h[r],X_train,y_train, description, metadata, y_dict, ending, question,model)
            print(f'The current error rate is {error_rate}')
            epsilon[r] = sum([w[i]*error_list[i] for i in range(N)])/sum([w[i] for i in range(N)]) #mark
        alpha[r] = math.log((1-epsilon[r])/epsilon[r])+math.log(K-1)
        for i in range(N):
            w[i] = w[i]*math.exp(alpha[r]*(1-error_list[i])) #mark
        w = normalize([w])[0]
        # print(error_list)
    error_rate, error_list = error_cal(h[r], X_test, y_test, description, metadata, y_dict, ending, question)
    print(f'final error rate: {error_rate}')
    return h, alpha, error_rate, error_list

def Summary(Xs,Ys,X,y,metadata,summaryprompt,description,y_dict,ending,question,model):
    sum = []
    errors = []
    prompts = []
    # Xs, Ys = ClusterSampling(X, y, len(set(y)), [1.0 / len(X), ] * len(X), 20)
    for x in tqdm(Xs):
        prompt = f"Task: {metadata}\nExample: {x}\nRequirement: {summaryprompt}\n"
        prompts.append(prompt)
        # print(prompt)
    summaries = generate_osllm_batch(prompts,model)
    for summary in summaries:
        # print(summary)
        # sum.append(summary)
        error_rate,_ = error_cal(summary,X[:20],y[:20], description, metadata, y_dict, ending, question,model)
        # e,_ = error_cal(summary,X,Y)
        # print(e)
        errors.append(error_rate)
    # r = errors.index(min(errors))
    res = sorted(range(len(errors)), key=lambda sub: errors[sub])[:3]
    print(f'The lowest error on the validation set is {min(errors)}')
    return 'Example: '.join([sum[r]+'\n' for r in res])

def generate(prompt):
    conversation = [{"role": "system", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=model_version,
            messages=conversation
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)


def generate_osllm(prompt, model, max_new_tokens=4096, temperature=0.2):
    """ completion by open-source llms with EAS
    Support models:
        - llama3-8b: llama-3-8b-instruct
        - mistral-7b: mistral-7b-instruct-v0.2
        - gemma-2-27b-it (in the near future)
    """
    request = {
        "sys_input": "You are a helpful assistant.",
        "input": prompt
    }
    
    llm_response = invoke_eas(
        request, model, input_col="input", output_col="output", 
        max_new_tokens=max_new_tokens, temperature=temperature, system_prompt= "you are a helpful assistant"
    )
    return llm_response["output"]


def generate_osllm_batch(prompts, model,  num_threads=1, max_tokens=4096, temperature=0.2, num_invokes=0):
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
    num_completed = 0
    ts = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(invoke_func_with_defaults, prompt) for prompt in prompts]
        for future in as_completed(futures):
            result = future.result()
            print(result)
            results.append(result)
    return results
            # json.dump(result, fout, ensure_ascii=False)
    #         fout.write('\n')
    #         fout.flush()
    #
    #         num_completed += 1
    #         if num_completed % 200 == 0:
    #             print(f"Complete {num_completed}/{len(prompts)} prompts in {time.time() - ts:.1f} secs.", flush=True)
    #


def error_cal(summary, X, y, description, metadata, y_dict, ending, question,model):
    error = 0
    elist = []
    a_list = []
    prompts = []
    for k,query in tqdm(enumerate(X)):
        query = query.split('###')[0]
        prompt = f"{description}\n{metadata}\nExample: {summary}\nNow here's the question: {query}\n{question}\n{ending}"
        prompts.append(prompt)
        # print(prompt)
    results = generate_osllm_batch(prompts,model)
    for s in results:
        try:
            answer = y_dict[s]
        except:
            if len(y_dict)==2:
                if s.lower().find('yes')>=0 or s.lower().find('consent')>=0 or s.lower().find('return')>=0 or s.lower().find('continue')>=0:
                    answer = True
                elif s.lower().find('no')>=0 or s.lower().find('avoid')>=0:
                    answer = False
                else:
                    print(s)
                    answer='Not sure'
            else:
                answer = 'Not sure'
        a_list.append(answer)
        if str(answer)!=str(y[k]):
            error += 1
            elist.append(1)
        else:
            elist.append(0)
    return error/float(len(X)), elist

def read_data(address):
    import json
    return [json.loads(i.strip('\n')) for i in open(address).readlines()]

def read_label(address):
    import pandas as pd
    df = pd.read_csv(address)
    label = df[df.columns[-1]].tolist()
    return label

def get_label_ratio(y):
    from collections import Counter
    y_c = Counter(y)
    return {c:y_c[c]/float(len(y)) for c in y_c}

def GPTEmbedding(X, k,portion='64',run=0,dataset='diabetes',model="text-embedding-ada-002"):
    X = [i.replace("\n", " ") for i in X]
    file_name = f'embeddings/embedding_{dataset}_{portion}_{str(run)}_'+str(k)+'.pkl'
    if not os.path.exists(file_name):
        embed = [client.embeddings.create(input = [i], model=model).data[0].embedding for i in tqdm(X)]
        with open(file_name,'wb') as f:
            pickle.dump(embed,f,protocol=1)
    else:
        embed = pickle.load(open(file_name,'rb'))
    return embed


# raw_files = []
data_file_address = '../datasets_serialized/'
# data_file_address = 'blood_raw_test.jsonl'
# raw_file_address = '/Users/yuanyifei/Downloads/blood/blood_test.csv'
data_files = os.listdir(data_file_address)
meta_file_address = 'task_instructions.json'
portion = '32'
model = 'llama3-8b'
if __name__ == '__main__':
    meta_info = json.load(open(meta_file_address))
    description_data = open('task_description.txt').readlines()
    description_data = {i.split(':')[0]: i.split(':')[1].strip('\n').lstrip(' ') for i in description_data}
    # Full sample test
    data_files = ['bank']
    for i in range(len(data_files)):
        print(data_files[i])
        metadata = description_data[data_files[i]]
        description,y_dict,summaryprompt,ending,question = read_meta(meta_info[data_files[i]])
        print("[[meatadata]]", metadata)
        print("[[description]]", description)
        print("[[y_dict]]", y_dict)
        print("[[summaryprompt]]", summaryprompt)
        print("[[ending]]", ending)
        print("[[question]]", question)
        for j in range(5):
            X_train = read_data(os.path.join(data_file_address,f'{data_files[i]}/cv{str(j)}/{data_files[i]}_cv{str(j)}train_{portion}.jsonl'))
            y_train = read_label(os.path.join(data_file_address,f'{data_files[i]}/cv{str(j)}/{data_files[i]}_tabllm_cv{str(j)}train_{portion}.csv')) #bank_tabllm_cv0test.csv
            X_test = read_data(os.path.join(data_file_address,f'{data_files[i]}/cv{str(j)}/{data_files[i]}_cv{str(j)}test.jsonl'))
            y_test = read_label(os.path.join(data_file_address,f'{data_files[i]}/cv{str(j)}/{data_files[i]}_tabllm_cv{str(j)}test.csv'))
            r = get_label_ratio(y_train)
            T = 1
            s = 4
            _, _, error_rate, error_list = Summaryboost(X_train, y_train, X_test, y_test, T, s, r, metadata, summaryprompt, description, y_dict, ending, question,portion=portion,run=j,dataset=data_files[i],model=model)
            output_address = os.path.join(data_file_address, f'{data_files[i]}/cv{str(j)}/summaryboost_{data_files[i]}_cv{str(j)}test_{portion}_{model}.pkl')
            with open(output_address,'wb') as f:
                pickle.dump({'error_rate':error_rate,'error_list':error_list},f,protocol=1)


    # Test one sample
    # X = read_data(data_file_address)
    # y = read_label(raw_file_address)
    # r = get_label_ratio(y)
    # T = 1
    # s = 16
    # Summaryboost(X, y, T, s, r)


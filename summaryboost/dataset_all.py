import pandas as pd
from main import generate
from tqdm import tqdm
import json
import os
import argparse

file_folder = '../datasets_serialized'
# file_address = '/Users/yuanyifei/Downloads/blood/blood_raw.csv'
# output_address = 'blood_raw_sum.jsonl'
threshold1, threshold2, threshold3 = 0. , 0. , 0.

def DataPreprocessor(file_address):
    df = pd.read_csv(file_address)
    ndf = df.to_numpy()
    col = df.columns
    y = {str(col[-1]):df[col[-1]]}
    text = []
    for r in ndf:
        text.append('\n'.join([col[i]+': '+str(r[i]) for i in range(len(r)-1)])) #add threshold
    return text,y


def Preprocessor(X,y,p_metadata,y_dict):
    # turn the structured tabular data into textual representation
    # p_metadata = """Data taken from the Blood Transfusion Service Center in Hsin-Chu City in Taiwan - this is a classification problem. The goal is to predict whether a given individual will consent or avoid donating blood. It includes the regency - months since last donation, frequency - total number of donation, Monetary - total blood donated in c.c., and Time - months since first donation."""
    # p_metadata = """The dataset aims to predict whether a blood donator returned for another donation. It includes the regency, frequency for blood donation."""
    answers = []
    for k,x in tqdm(enumerate(X)):
        p_prompt = f"""{p_metadata}
        Here is one example from this dataset.
        Goal: Describe the given data in words.
        {x}
        Use your creativity to describe this data accurately and concisely. Do not add any additional information."""
        a = generate(p_prompt)
        conclusion = f'###\nHence this {list(y.keys())[0]} was {y_dict[list(y.values())[0][k]]}.'
        answers.append(a+conclusion)
    return answers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some arguments')
    parser.add_argument('portion',type=str,default='32',choices=['32','64','128','256','all'])
    parser.add_argument('category',type=str,default='diabetes',choices=['bank','blood','calhousing','car','creditg','diabetes','heart','income','jungle'])

    meta_data = open('task_description.txt').readlines()
    # print(meta_data)
    meta_data = {i.split(':')[0]:i.split(':')[1].strip('\n').lstrip(' ') for i in meta_data}
    portion = '32'
    for category in ['car']:
        print(category)
        if category != 'car':
            y_dict = {True: 'Yes', False: 'No'}
        else:
            y_dict = {0: 'Unacceptable', 1: 'Acceptable', 2: 'Good', 3: 'Very Good'}
        for j in range(5):
            print(j)
            # X,y = DataPreprocessor(os.path.join(file_folder,category,f'cv{str(j)}',f'{category}_tabllm_cv{str(j)}test.csv'))
            X,y = DataPreprocessor(os.path.join(file_folder,category,f'cv{str(j)}',f'{category}_tabllm_cv{str(j)}train_{portion}.csv'))
            answers = Preprocessor(X,y,meta_data[category],y_dict)
            # with open(os.path.join(os.path.join(file_folder,category,f'cv{str(j)}'),category+f'_cv{str(j)}test.jsonl'),'w') as f:
            with open(os.path.join(os.path.join(file_folder,category,f'cv{str(j)}'),category+f'_cv{str(j)}train_{portion}.jsonl'),'w') as f:
                for a in answers:
                    f.write(json.dumps(a)+'\n')
            # print(answers)







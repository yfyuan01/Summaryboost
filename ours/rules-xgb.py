# -*- coding: utf-8 -*-
# 
# Mine mini-group rules with LR or SVC.
# 
# cmds:
#   python rules.py --dataset bank --cv 0 --gmethod svc

import argparse
import copy
import os
import numpy as np
import pandas as pd


from collections import Counter
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from xgboost import XGBClassifier, DMatrix

from evaluation import trans_pred_labels,precision_recall_fscore_support
from helper.utils import load_base_instructions, sink_to_jsonl, load_jsonl_with_keys
from helper.serilization import serialize_by_tabllm
from llm_apis.common_api import determine_invoke_func, batch_invoke_llm, parallel_invoke_llm
from selection_strategy import sv_select_max_abs, sv_select_min

from constants import TABLLM_DATASETS, TABLLM_BINARY_DATASETS, INSTANCE_SEP


# dtype_cnter = Counter()
ONE_HOT_DTYPES = [object, pd.BooleanDtype]
LR_HPS = {
    "penalty": ["l1", "l2"],
    "C": [1e5, 1e4, 1e3, 1e2, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  # inverse of regularization strength
}

""" The parameter C, common to all SVM kernels, trades off misclassification of training examples 
against simplicity of the decision surface. A low C makes the decision surface smooth, 
while a high C aims at classifying all training examples correctly. 

gamma defines how much influence a single training example has. 
The larger gamma is, the closer other examples must be to be affected.    
"""
SVC_HPS = {
    "C": [1e4, 1e3, 1e2, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4],
    "gamma": [1e4, 1e3, 1e2, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4],
}


SV_SELECTION_PROMPT = """\
You are now required to select a subset of examples from the above as support vectors to aid prediction (as in SVM models).

Instructions for Selecting Support Vectors:
1. Objective: Select a subset of examples from the provided dataset that will effectively serve as support vectors to improve the performance of our classification model.
2. Representativity: Ensure the selected examples are representative of the entire dataset. This should include a broad and diverse range of classes, feature values, and any relevant distributions.
3. Diversity: Choose examples that capture the diversity within the dataset. Consider examples from all classes and across different feature values to ensure a wide coverage.
4. Boundary Cases: Prioritize selecting examples near the class decision boundaries, where the distinctions between classes are less clear. These examples are often critical in defining the decision surface.
5. Relevance to Model Performance: Select examples that have a significant impact on the model’s learning process. Focus on those that help to minimize classification errors and improve overall accuracy.
6. Balanced Sampling: If the dataset is imbalanced, ensure that the subset reflects a balanced representation of all classes to avoid bias in model training.
7. Specific Features: Include a range of values for key predictive features identified in the dataset. Ensure that critical features are well-represented in the selected examples.
8. Efficiency: Aim for a selection that balances performance improvement with computational efficiency. The subset should be **as small as possible** while still retaining its effectiveness as support vectors.

Requirements:
You are given a STRICT limit on the number of selected support vectors, say 16. 
ALL classes are properly represented by support vectors.
Only retain CRITICAL features that distinguish support vectors from the rest.

Output format:
SV 1
[[label]] {{the label of the example}}
[[critical feature]] {{the critical features of the example}}

SV 2
...
"""

def load_training_data(data_dir:str, dataset:str, cv:int, task_instruction:dict, num_examples:int, usage:str):
    print(f"load_training_data with dataset={dataset}  cv={cv}  num_examples={num_examples}")
    question = task_instruction["question"] 
    answer_choices = task_instruction["answer_choices"]
    answer_requirement = task_instruction["answer_requirement"]
    
    # load tabllm serilization
    path = os.path.join(data_dir, dataset, f"cv{cv}", f"{dataset}_tabllm_cv{cv}train_{num_examples}.csv")
    data = pd.read_csv(path, header=0)
    columns = list(data.columns)
    notes = []
    is_train = True if usage == "train" else False
    answer_cond = answer_choices if usage == "train" else answer_requirement
    for row in data.itertuples(index=False):
        notes.append((serialize_by_tabllm(tuple(row), columns, question, answer_cond, is_train=is_train),
                      row[-1]))
    
    # load raw features
    path = os.path.join(data_dir, dataset, f"cv{cv}", f"{dataset}_raw_cv{cv}train_{num_examples}.csv")
    data = pd.read_csv(path, header=0)
    
    features = data.iloc[:, :-1]
    label = data.iloc[:, -1].astype(int)

    numerical_cols = [col for col in features.columns if features[col].dtype not in ONE_HOT_DTYPES]
    non_numerical_cols = [col for col in features.columns if features[col].dtype in ONE_HOT_DTYPES]
    # print(numerical_cols, non_numerical_cols)

    """ apply z-value normalization on numerical features (int/float) and 
    one-hot encoding on non-numerical features (bool/object). 
    Get two numpy matrices and concatenate then together as LR features.
    """
    feature_zval = StandardScaler().fit_transform(features[numerical_cols]) \
        if numerical_cols else None
    features_ohe = pd.get_dummies(features[non_numerical_cols], dtype=float).values \
        if non_numerical_cols else None
    if feature_zval is None:
        features_norm = features_ohe
    elif features_ohe is None:
        features_norm = feature_zval
    else:
        features_norm = np.concatenate([feature_zval, features_ohe], axis=1)
    
    print(f"feature shape raw {features.shape}  vs. lr {features_norm.shape}")
    print()
    return features, features_norm, label, notes



def get_info_entropy(proba:np.array):
    result = np.zeros((proba.shape[0],))
    for class_i in range(proba.shape[1]):
        result -= proba[:, class_i] * np.log(proba[:, class_i])
    return result


def modelling_entropy(features:pd.DataFrame, label: pd.DataFrame):
    '''
    Args:
        features: pd.DataFrame, the training tabular data
        label: pd.DataFrame, the label
    return: 
        extracted_train_info: List[dict]
    '''
    # Preprocessing data
    # Encode for categorical data
    le = LabelEncoder()
    
    for feat_name in features.columns:
        if type(features.loc[0, feat_name]) is not int \
        and type(features.loc[0,feat_name]) is not bool:
            features[feat_name] = le.fit_transform(features[feat_name])

    X_train = features[[feat_name for feat_name in features.columns]]
    y_train = label
    
    dtrain = DMatrix(X_train, label=y_train, weight=np.ones(len(y_train)))  # 示例中简单地为每个样本赋予相同的权重1
    
    # Intialize XGBoost Classifier
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    
    # Train XGBoost Classifier
    model.fit(X_train, y_train)
    
    # Obtain prediction probabilities on the training set
    predictions_proba = model.predict_proba(X_train)  # 获取正类的概率

    # Obtain information entropy for training samples
    entropy = get_info_entropy(predictions_proba)

    # Group training samples using their leaf index on the first booster of xgboost
    group_ids = model.get_booster().predict(dtrain, iteration_range = (0,1), pred_leaf=True)

    extracted_train_info = [
            {
                "index": int(i), 
                "label": int(label[i]), 
                "coef": float(xg_results[0]),
                "group_id": int(xg_results[1]),
                "pred_proba": float(xg_results[2])
            } 
            for i, xg_results in enumerate(zip(entropy, group_ids, predictions_proba[:,1]))
        ]
    return extracted_train_info

def grouping_with_llm(train_info: list, train_notes:list, trend_cmd:str, sum_cmd:str, llm_model:str = "gpt-4-turbo"):
    '''New grouping procedure using meta information provided by xgboost.
    Args:
        train_info: list. The format is identical to the return value of modelling_entropy().
        train_notes: list. Each element of train_notes is Tuple(str, str). The first element is the text note. The second element is the label
        trend_cmd:str. the prompt for obtaining trends from training samples which is available at task_instructions_formal.json
        sum_cmd:str. the prompt for summarizing trends in differnt groups which is available at task_instructions_formal.json
        llm_model:str. the model for generating trends and summarizing trends to the meta rule.
    Return:
        results: list. The first (n-1) elements are trends summarized by llm_model. The last element is the meta rule summarized from trends by llm_model.
    '''
    # Step 1. group training samples according to group_id
    training_groups = {}
    for elem in train_info:
        group_id = elem["group_id"]
        training_groups[group_id] = training_groups.get(group_id, [])
        training_groups[group_id].append(copy.deepcopy(elem))

    # Step 2. obtain the llm_model's extracted rules 
    batch_prompts = []
    for group_id in training_groups.keys():
        current_group = training_groups[group_id]
        # print(f"current_group = {current_group}")
        in_group_notes = [train_notes[elem['index']][0] for elem in current_group]
        num_pos = sum([train_notes[elem['index']][1] for elem in current_group])
        num_neg = sum([1 - train_notes[elem['index']][1] for elem in current_group])
        prompt = INSTANCE_SEP.join(in_group_notes + [trend_cmd])
        batch_prompts.append({
            'cmd': 'trend',
            'input': prompt,
            'num_pos': num_pos,
            'num_neg': num_neg,
            'sample_indices': [elem['index'] for elem in current_group]
        })

    # Step 3. Invoke the llm_model to obtain summarized rules
    results = batch_invoke_llm(batch_prompts, llm_model, num_threads=16, temperature=0.2, max_tokens=4094)

    trends = [elem['output'] for elem in results]

    # Step 4. Summarized rules
    llm_api = determine_invoke_func(llm_model, temperature=0.2, max_tokens=4094)
    
    prompt = INSTANCE_SEP.join(trends + [sum_cmd])
    summarization = llm_api({
        "cmd": "summarization",
        "input": prompt,
    })


    results.append(summarization)
    
    return results

def boost_grouping(train_info:list, meta_rule:str, train_notes:list, train_eval_notes:list, 
                   labels: list, task_meta:dict, llm_model:str, num_shots:int,
                   dataset_dir:str, dataset:str, sink_suffix:str):
    '''One iterating step for boosting rules.
    Args:
        train_info: list. Training data distribution information extracted by XGBoost.
            The format is identical to the return value of modelling_entropy().
        meta_rule: str. The original rule
        train_notes: list. Each element of train_notes is Tuple(str, str). The first element is the text note with  the correct answer. The second element is the label.
        train_eval_notes: list. Each element of train_notes is Tuple(str, str). The first element is the text note without the correct answer. The second element is the label.
        labels: list. each element is a label value corresponding to the question.
        task_meta: dict. It is equivalent to an element of task_instructions_formal.json.
        llm_model: str. see grouping_with_llm().
        ...
    Return:
        summarization_for_error: list. The structure is the same as the return value of grouping_with_llm().
    '''
    task_title = task_meta["title"]
    task_description = task_meta["description_brief"]
    answer_choices = task_meta["answer_choices"]
    eval_fname = f"{sink_suffix}-{llm_model}"
    average_method = "binary" if dataset != "car" else "macro"
    trend_cmd = task_meta["trend"]
    sum_cmd = task_meta["summarization"]
    
    # Step 1: Select 50% eval samples with highest entropy. The number should not exceed 128
    select_highest_entropy = sv_select_max_abs(train_info, num_shots = min(int(len(train_info)/2), 128))
    notes_highest_entropy = [train_eval_notes[i] for i, _ in select_highest_entropy]
    labels_highest_entropy = [labels[i] for i, _ in select_highest_entropy]

    # Step 2: Select num_shots training samples with lowest entropy
    select_lowest_entropy = sv_select_min(train_info, num_shots = num_shots)
    notes_lowest_entropy = [train_notes[i] for i, _ in select_lowest_entropy]

    # Step 3: Evaluate our method on the 50% samples with highest entropy and collect the error list
    prompts = []
    num_proc = 0
    total_num_tokens = 0
    for eval_note, eval_label in zip(notes_highest_entropy, labels_highest_entropy):
        patterns = f"Useful patterns for the task at hand:\n{meta_rule}"
        prompt_shot_examples = INSTANCE_SEP.join([_x[0] for _x in notes_lowest_entropy])
        prompt = f'''{task_title}
{task_description}

{patterns}

###

[FEW-SHOT EXAMPLES START]

{prompt_shot_examples}

[FEW-SHOT EXAMPLES END]

###

[CURRENT QUESTION START]

{eval_note[0]}'''
        total_num_tokens += len(prompt) // 4
        num_proc += 1
        if num_proc == 1:
            print(prompt)
        prompts.append({
            "id": dataset + str(num_proc).zfill(5),
            "input": prompt,
            "label": eval_note[1],
            "answer": answer_choices[eval_note[1]]
        })
    print(f"# prompts: {len(prompts)}, # token: {total_num_tokens}, avg {total_num_tokens//num_proc}")
    parallel_invoke_llm(prompts, 
                        model=llm_model, 
                        outfile=os.path.join(dataset_dir, eval_fname),
                        num_threads=16, 
                        temperature=0.2, 
                        max_tokens=128,
                        num_invokes=0)
    y_true, y_pred, d_size, err_ratio, idx_list = trans_pred_labels(
        records = load_jsonl_with_keys(dataset_dir, eval_fname, keys = None, silent = True),
        dataset = dataset
    )
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average_method)
    print(f"At the current step, f1: {f1}")
    error_list = [idx_list[i] for i in range(len(y_true)) if y_pred[i]!=y_true[i]]
    print(f'{len(error_list)}/{len(notes_highest_entropy)} error samples are extracted from feedback.')

    # TODO Step 4: summarize on error samples.
    train_info_for_error = [train_info[i] for i in error_list]
    train_notes_for_error = [train_notes[i] for i in error_list]
    for i in range(len(train_info_for_error)):
        train_info_for_error[i]['original_index'] = train_info_for_error[i]['index']
        train_info_for_error[i]['index'] = i

    summarization_for_error = grouping_with_llm(
        train_info = train_info_for_error,
        train_notes = train_notes_for_error,
        trend_cmd = trend_cmd,
        sum_cmd = sum_cmd
    )
    print(meta_rule)
    print('=========')
    print(summarization_for_error[-1]['output'])
    return summarization_for_error
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets_serialized", help="name of dataset to process.")
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset to process If dataset = all, then run for all binary tabular datasets.")
    parser.add_argument("--cv", type=int, default=0, help="cross validation number If cv = -1, then run for all cross validation datasets.")
    parser.add_argument("--num_examples", type=str, default='128', help="number of training examples")
    parser.add_argument("--group_size", type=int, default=16, help="number of training sample groups to be analyzed by a superior LLM, which is equivalent to num_shots in main.py. Default = 16")
    parser.add_argument("--gmethod", required=True, choices=["lr", "svc", "llm4sv", "xgboost"], help="example grouping method")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    print("End of arg parsing.\n\n")

    datasets = TABLLM_BINARY_DATASETS if args.dataset == "all" else [args.dataset]
    instructions = load_base_instructions("task_instructions_formal.json") 
    llm4sv_selections = []
    cvs = [args.cv] if args.cv >=0 else [0,1,2,3,4]
    for dataset in datasets:
        for this_cv in cvs:
        # load training data w.r.t dataset&cv&num_examples
            raw_fts, norm_fts, label, train_notes = load_training_data(
                args.data_dir, 
                dataset,
                this_cv, 
                instructions[dataset],
                args.num_examples, 
                usage = "train"
            )

            _, _, _, train_eval_notes = load_training_data(
                args.data_dir, 
                dataset,
                this_cv, 
                instructions[dataset],
                args.num_examples, 
                usage = "test"
            )
            
            print(train_notes[0])
            print(train_eval_notes[0])

            if args.gmethod == "xgboost":
                # ** Step 1: fit a XGBoost model on the training sample **
                # 1.1 Fit the model
                # 1.2 Select important features
                # 1.3 group samples using the first tree of XGBoost
                
                ent_outputs = modelling_entropy(raw_fts, label)
                sink_to_jsonl(
                    data_dir=os.path.join(args.data_dir, dataset, f"cv{this_cv}"),
                    filename=f"{dataset}-xgboostentropy-cv{this_cv}-e{args.num_examples}",
                    data=ent_outputs
                )

                # # ** Step 2: use a superior LLM to summarize rules for each group
                # # 2.1 Summarize rules for each group
                rules = grouping_with_llm(
                    train_info = ent_outputs,
                    train_notes = train_notes,
                    trend_cmd = instructions[dataset]["trend"],
                    sum_cmd = instructions[dataset]["summarization"])

                additional_rules = boost_grouping(
                    train_info = ent_outputs, 
                    meta_rule = rules[-1]['output'], 
                    train_notes = train_notes, 
                    train_eval_notes = train_eval_notes, 
                    labels = label, 
                    task_meta = instructions[dataset],
                    num_shots = 16,
                    dataset_dir = args.data_dir, 
                    dataset = dataset, 
                    llm_model = "gpt-4-turbo",
                    sink_suffix = f"{dataset}-rulesfeedback{args.gmethod}-cv{this_cv}-e{args.num_examples}")

                sink_to_jsonl(
                    data_dir = os.path.join(args.data_dir, dataset, f"cv{this_cv}"),
                    filename = f"{dataset}-rulesxgboost-cv{this_cv}-e{args.num_examples}-automated",
                    data = rules
                )

                sink_to_jsonl(
                    data_dir = os.path.join(args.data_dir, dataset, f"cv{this_cv}"),
                    filename = f"{dataset}-additionalrulesxgboost-cv{this_cv}-e{args.num_examples}-automated",
                    data = additional_rules
                )
            print("========\n\n")
    

if __name__ == "__main__":
    main()

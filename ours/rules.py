# -*- coding: utf-8 -*-
# 
# Mine mini-group rules with LR or SVC.
# 
# cmds:
#   python rules.py --dataset bank --cv 0 --gmethod svc

import argparse
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

from helper.utils import load_base_instructions, sink_to_jsonl
from helper.serilization import serialize_by_tabllm
from llm_apis.common_api import determine_invoke_func, batch_invoke_llm

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

def load_training_data(data_dir, dataset, cv, task_instruction, num_examples, usage):
    print(f"load_training_data with dataset={dataset}  cv={cv}  num_examples={num_examples}")
    question = task_instruction["question"] 
    answer_choices = task_instruction["answer_choices"]
    
    # load tabllm serilization
    path = os.path.join(data_dir, dataset, f"cv{cv}", f"{dataset}_tabllm_cv{cv}train_{num_examples}.csv")
    data = pd.read_csv(path, header=0)
    columns = list(data.columns)
    notes = []
    is_train = True if usage == "train" else False
    for row in data.itertuples(index=False):
        notes.append((serialize_by_tabllm(tuple(row), columns, question, answer_choices, is_train=is_train), row[-1]))
    
    # load raw features
    path = os.path.join(data_dir, dataset, f"cv{cv}", f"{dataset}_raw_cv{cv}train_{num_examples}.csv")
    data = pd.read_csv(path, header=0)
    
    features = data.iloc[:, :-1]
    label = data.iloc[:, -1].astype(int)
    ### print data details
    # print(data)
    # for column_name in data.columns:
    #     print(column_name, data[column_name].dtype, data[column_name].values[:10])
    #     dtype_cnter[data[column_name].dtype] += 1
    # print()

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


def modeling(features, label, binary_task, gmethod):
    """基于传统机器学习方法（gmethod, 目前是logistic regression和support vector classification），
    对当前数据 (fetures&label)建模，返回结果。
    modeling相当于是rule-mining的一部分，由rule.main调用，modeling函数的返回结果直接用于规则生成。

    return:
        -gmethod=lr时，返回lr模型对样本（features）的预测值，后续基于预测结果进行样本分组
        -gmethod=svc时，返回svc模型下各样本的支持向量系数以及模型预测值，后续基于支持向量系数进行in-context样本的筛选，
            模型预测值暂时没有使用
    """
    if gmethod == "lr":
        estimator = LogisticRegression(
            class_weight="balanced", fit_intercept=True, solver="liblinear", verbose=False, max_iter=200
        )
        hps = LR_HPS
    elif gmethod == "svc":
        estimator = SVC(kernel='rbf', class_weight="balanced", verbose=False, probability=True)
        hps = SVC_HPS

    metric = "roc_auc" if binary_task else "balanced_accuracy"  # roc_auc_ovo
    clf = GridSearchCV(
        estimator=estimator, param_grid=hps, cv=StratifiedKFold(n_splits=5), 
        scoring=metric, n_jobs=40, verbose=0
    )
    clf.fit(features, label)
    print(clf.best_params_, clf.best_score_)
    model = clf.best_estimator_
    
    if gmethod == "svc":
        support_vectors = model.support_vectors_
        support_vector_indices = model.support_
        support_vector_coefficients = model.dual_coef_
        sv_outputs = [
            {
                "index": int(i), 
                "label": int(label[i]), 
                "coef": round(support_vector_coefficients[0][n], 3)
            } 
            for n, i in enumerate(support_vector_indices)
        ]
        print("support_vectors:", support_vectors.shape)
        print("support_vector_indices:", support_vector_indices.shape)
        print("support_vector(index, label, coef):", 
              [(i, label[i], round(support_vector_coefficients[0][n], 3)) 
               for n, i in enumerate(support_vector_indices) ])
    else:
        sv_outputs = None

    y_pred = model.predict(features) if not binary_task else model.predict_proba(features)[:, 1]
    # instances = [(round(y_pred[i], 3), label[i]) for i in range(y_pred.shape[0])]
    # instances.sort(key=lambda x: x[0])
    # print(instances)
    my_metric = roc_auc_score(label, y_pred, average="macro") if binary_task \
        else balanced_accuracy_score(label, y_pred)    
    print(f"My {metric} is {my_metric}")

    scores = [(i, yi) for i, yi in enumerate(y_pred)]
    scores.sort(key=lambda x: x[1])
    return scores, sv_outputs


def modelling_entropy(features:pd.DataFrame, label: pd.DataFrame):
    '''
    Args:
        features: pd.DataFrame, the training tabular data
        label: pd.DataFrame, the label
    return: 
        entropy: list(tuple). the first element is the data id. The second element is the data's entropy
            calculated by xgboost.
    '''
    # 数据预处理
# 对类别型特征进行编码
    le = LabelEncoder()
    
    for feat_name in features.columns:
        if type(features.loc[0, feat_name]) is not int \
        and type(features.loc[0,feat_name]) is not bool:
            features[feat_name] = le.fit_transform(features[feat_name])

    X_train = features[[feat_name for feat_name in features.columns]]
    y_train = label
    
    dtrain = DMatrix(X_train, label=y_train, weight=np.ones(len(y_train)))  # 示例中简单地为每个样本赋予相同的权重1
    
    # 初始化XGBoost模型
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    predictions_proba = model.predict_proba(X_train)[:, 1]  # 获取正类的概率

    entropy = - predictions_proba * np.log(predictions_proba) \
              - (1 - predictions_proba) * np.log((1 - predictions_proba))

    ent_outputs = [
            {
                "index": int(i), 
                "label": int(label[i]), 
                "coef": float(ent)
            } 
            for i, ent in enumerate(entropy)
        ]
    return ent_outputs



def grouping_instructions(scores, notes, trend_cmd, sum_cmd, group_size, steps, llm_model="gpt-4-turbo"):
    '''
    基于模型预测值（scores）进行样本分组，每个组的样本先抽取trend，所有组的trend汇总后进行总结，作为rule。

    Params:
        - scores: 样本的预测值，当前只关注binary任务，因此预测值是样本被分类为正样本的概率
        - notes: tabular样本的NLP表示，目前采用tabllm的序列化方法，格式为 The attr_name is value. ...
        - trend_cmd: 组样本抽取trend的prompt，见task_instruction.json
        - sum_cmd: 所有组的trend汇总的prompt，见task_instruction.json
        - group_size: 一个组包含样本的个数
        - steps: 从1个组切换到下1个组的index增量，steps=group_size表示各个组互不重合，steps<group_size时组之间样本由重合
        - llm_model: 具体执行trend_cmd&sum_cmd的llm

    Return: List[dict], each dict contains the input/output of trend/sum request.
    '''
    assert (len(notes) - group_size) % steps == 0,\
        f"Error group_size and steps config: {len(notes)} {group_size} {steps}"
    
    i = 0
    trend_prompts = []
    while i < len(scores):
        in_group_notes = [notes[i][0] for i, _ in scores[i: i+group_size]] 
        prompt = INSTANCE_SEP.join(in_group_notes + [trend_cmd])
        num_pos = sum(notes[i][1] for i, _ in scores[i: i+group_size])
        num_neg = sum(1 - notes[i][1] for i, _ in scores[i: i+group_size])
        trend_prompts.append({
            "cmd": "trend",
            "input": prompt,
            "start": i,
            "end": i + steps - 1,
            "num_pos": num_pos,
            "num_neg": num_neg,
        })
        print(f"start/end={i}/{i + steps - 1}, pos={num_pos}, neg={num_neg}")
        i += steps
    
    results = batch_invoke_llm(trend_prompts, llm_model, num_threads=16, temperature=0.2, max_tokens=4094)

    trends = [(x["start"], x["output"]) for x in results]
    trends.sort(key=lambda x: x[0])
    llm_api = determine_invoke_func(llm_model, temperature=0.2, max_tokens=4094)
    
    prompt = INSTANCE_SEP.join([x[1] for x in trends] + [sum_cmd])
    results.append(llm_api({
        "cmd": "summarization",
        "input": prompt,
    }))
    return results


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
    instructions = load_base_instructions("task_instructions.json") 
    llm4sv_selections = []
    cvs = args.cv if args.cv >=0 else [0,1,2,3,4]
    for dataset in datasets:
        for this_cv in cvs:
        # load training data w.r.t dataset&cv&num_examples
            usage = "train" if args.gmethod in ["lr", "svc"] else "sv"
            raw_fts, norm_fts, label, notes = load_training_data(
                args.data_dir, 
                dataset,
                this_cv, 
                instructions[dataset],
                args.num_examples, 
                usage
            )
    
            if args.gmethod in ["lr", "svc"]:
                # gmethod=lr: 1. scores by modeling; 2. groupwise trand&sum as rule by grouping_instructions
                # gmethod=svc: 1. scores&sv_outputs by modeling; 
                #       2. sink sv_outputs (i.e., sv coef) for sample selection in main.py
                #       3. groupwise trand&sum as rule by grouping_instructions
                binary_task = True if len(np.unique(label)) == 2 else False
                scores, sv_outputs = modeling(norm_fts, label, binary_task, args.gmethod)
    
                # ---debug ---
                # print(f"scores = {scores}")
                # print(f"sv_outputs = {sv_outputs}")
                # print(f"#sv_outputs = {len(sv_outputs)}")
                # print(f"notes = {notes}")
                # print(f"#notes = {len(notes)}")
                # return
                # ---debug---
    
    
                if sv_outputs is not None:
                    sink_to_jsonl(
                        data_dir=os.path.join(args.data_dir, dataset, f"cv{this_cv}"),
                        filename=f"{dataset}-svcoef-cv{this_cv}-e{args.num_examples}",
                        data=sv_outputs
                    )
                
                prompts = grouping_instructions(scores, notes, 
                                                trend_cmd=instructions[dataset]["trend"], 
                                                sum_cmd=instructions[dataset]["summarization"], 
                                                group_size=args.group_size, steps=args.group_size)
                sink_to_jsonl(
                    data_dir=os.path.join(args.data_dir, dataset, f"cv{this_cv}"),
                    filename=f"{dataset}-rules{args.gmethod}-cv{this_cv}-e{args.num_examples}-s{args.group_size}",
                    data=prompts
                )
            elif args.gmethod == "xgboost":
                ent_outputs = modelling_entropy(raw_fts, label)
                sink_to_jsonl(
                    data_dir=os.path.join(args.data_dir, dataset, f"cv{this_cv}"),
                    filename=f"{dataset}-xgboostentropy-cv{this_cv}-e{args.num_examples}",
                    data=ent_outputs
                )
                
            elif args.gmethod == "llm4sv":
                # using llm to directly select samples as support vectors, following SV_SELECTION_PROMPT
                # Here we generate prompt for each dataset first.
                notes.sort(key=lambda x: x[1])
                input_elements = [instructions[dataset]["description"]] + [
                    f"Example {ni+1}\n{note}" for ni, (note, _) in enumerate(notes)
                ] + [SV_SELECTION_PROMPT]  # instructions["support_vector"]["selection"]
                llm4sv_selections.append({
                    "dataset": dataset,
                    "cv": this_cv,
                    "num_examples": args.num_examples,
                    "input": INSTANCE_SEP.join(input_elements)
                })
    
            print("========\n\n")
    
        # for dtype, cnt in dtype_cnter.most_common():
        #     print(dtype, cnt)
    
        # follow-up step for gmethod == "llm4sv", request gpt-4o for sv output. 
        if args.gmethod == "llm4sv":
            sink_to_jsonl(
                data_dir=args.data_dir, 
                filename=f"sv-selection-cv{args.cv}-e{args.num_examples}", 
                data=batch_invoke_llm(llm4sv_selections, "gpt-4o", num_threads=16, temperature=0.2, max_tokens=4096)
            )
    

if __name__ == "__main__":
    main()

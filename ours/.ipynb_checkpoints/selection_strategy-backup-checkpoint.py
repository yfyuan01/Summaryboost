# -*- coding: utf-8 -*-
import random

import numpy as np

def sv_select_max_abs(svcoef:list, num_shots:int):
    '''
    Select num_shots samples from their svm coefficient info
        with the strategy of maximum absolute value of coef.
    Args:
        svcoef:List[dict]. Key = "index" denotes te item id.
            Key = "coef" denotes the coefficient value.
        num_shots: the number of representative samples to be selected
    Return:
        select:List[Tuple]. The first element of each tuple
            denotes the item id. The second element of each
            tuple denotes the coefficient value.
    '''
    svcoef.sort(key=lambda x: x["coef"])
    select = []
    le = 0 # left index
    ri = len(svcoef) - 1 # right index
    
    # select samples with largest absolute value of svm coefficient
    while len(select) < num_shots and le < ri:
        if svcoef[le]["coef"] < 0:
            select.append((svcoef[le]["index"], svcoef[le]["coef"]))
            le += 1
        if svcoef[ri]["coef"] > 0:
            select.append((svcoef[ri]["index"], svcoef[ri]["coef"]))
            ri -= 1
    select.sort(key=lambda x: x[1])
    return select

def sv_select_min(svcoef:list, num_shots:int):
    '''
    Select num_shots samples from their svm coefficient info
        with the strategy of maximum absolute value of coef.
    Args:
        svcoef:List[dict]. Key = "index" denotes te item id.
            Key = "coef" denotes the coefficient value.
        num_shots: the number of representative samples to be selected
    Return:
        select:List[Tuple]. The first element of each tuple
            denotes the item id. The second element of each
            tuple denotes the coefficient value.
    '''
    svcoef.sort(key=lambda x: x["coef"])
    select = [(elem["index"], elem["coef"]) for elem in svcoef[:num_shots]]
    return select

def sv_select_balanced_max_abs(svcoef:list, num_shots:int):
    '''
    Select num_shots samples from their svm coefficient info
        with the strategy of maximum absolute value of coef.
    Args:
        svcoef:List[dict]. Key = "index" denotes te item id.
            Key = "coef" denotes the coefficient value.
        num_shots: the number of representative samples to be selected
    Return:
        select:List[Tuple]. The first element of each tuple
            denotes the item id. The second element of each
            tuple denotes the coefficient value.
    '''
    svcoef.sort(key=lambda x: x["coef"])
    select = []
    le = 0 # left index
    ri = len(svcoef) - 1 # right index
    
    # select samples with largest absolute value of svm coefficient
    while len(select) < num_shots and le < ri:
        if svcoef[le]["coef"] < 0 and svcoef[ri]["coef"] > 0:
            select.append((svcoef[le]["index"], svcoef[le]["coef"]))
            select.append((svcoef[ri]["index"], svcoef[ri]["coef"]))
        le += 1
        ri -= 1
    select.sort(key=lambda x: x[1])
    return select

def sv_select_systematic_sampling(svcoef:list, num_shots:int):
    '''
    Select num_shots samples from their svm coefficient info
        with the strategy of systematic sampling.
    Args:
        svcoef:List[dict]. Key = "index" denotes te item id.
            Key = "coef" denotes the coefficient value.
        num_shots: the number of representative samples to be selected
    Return:
        select:List[Tuple]. The first element of each tuple
            denotes the item id. The second element of each
            tuple denotes the coefficient value.
    '''
    svcoef.sort(key=lambda x: x["coef"])
    select = []
    gap = max(int(len(svcoef) / num_shots),1)
    print(f"num_shots={num_shots}")
    for current in range(0,len(svcoef), gap):
        print(f"current={current}")
        select.append((svcoef[current]["index"], svcoef[current]["coef"]))
    select.sort(key=lambda x: x[1])
    left = int((len(select) - num_shots)/2)
    select = select[left:(left+num_shots)]
    return select

def sv_select_balanced_systematic_sampling(svcoef:list, num_shots:int):
    '''
    Select num_shots samples from their svm coefficient info
        with the strategy of balanced systematic sampling.
    Args:
        svcoef:List[dict]. Key = "index" denotes te item id.
            Key = "coef" denotes the coefficient value.
        num_shots: the number of representative samples to be selected
    Return:
        select:List[Tuple]. The first element of each tuple
            denotes the item id. The second element of each
            tuple denotes the coefficient value.
    '''
    svcoef.sort(key=lambda x: x["coef"])
    select = []
    mid = 0
    n_all = len(svcoef)
    while mid < n_all and svcoef[mid]["coef"] <= 0:
        mid += 1
    neg_gap = int(2 * mid / num_shots)
    pos_gap = int(2 * (n_all - mid)/num_shots)
    print(f"num_shots={num_shots}")

    # select 50% from negative samples
    for current in range(0, mid, neg_gap):
        print(f"current={current}")
        select.append((svcoef[current]["index"], svcoef[current]["coef"]))
        if len(select) == int(num_shots / 2):
            break

    # select 50% from positive samples
    for current in range(mid, n_all, pos_gap):
        print(f"current={current}")
        select.append((svcoef[current]["index"], svcoef[current]["coef"]))
        if len(select) == num_shots:
            break
    select.sort(key=lambda x: x[1])
    return select

# Yifei: add the error sampling weights to the original selection algorithm
def sv_select_systematic_sampling_renew(svcoef:list, num_shots:int, error_idx:list, temperature=2):
    '''
    Select num_shots samples from their svm coefficient info
        with the strategy of systematic sampling.
    Args:
        svcoef:List[dict]. Key = "index" denotes te item id.
            Key = "coef" denotes the coefficient value.
        num_shots: the number of representative samples to be selected
        error_idx: index of the wrong prediction samples
        temperature: the temperature of to enlarge the error weight.
    Return:
        select:List[Tuple]. The first element of each tuple
            denotes the item id. The second element of each
            tuple denotes the coefficient value.
    '''
    svcoef.sort(key=lambda x: x["coef"])
    candidate = [(s['index'],s['coef'])for s in svcoef]
    gap = max(int(len(svcoef) / num_shots),1)
    print(f"num_shots={num_shots}")
    weight_dict = {i["index"]:0 for i in svcoef}
    for current in range(0,len(svcoef), gap):
        print(f"current={current}")
        weight_dict[svcoef[current]["index"]] = 1
        # select.append((svcoef[current]["index"], svcoef[current]["coef"]))
    for e in error_idx:
        if e in weight_dict:
            weight_dict[e]+=temperature
    candidate.sort(key=lambda x: x[0])
    weights = [weight_dict[c[0]] for c in candidate]
    weights = [w/sum(weights) for w in weights] #scale to make the total sum 1.0
    select = np.random.choice(range(len(candidate)), size=num_shots, p=weights, replace=False)
    select = [candidate[s] for s in select]
    return select

def sv_select_min_renew(svcoef:list, num_shots:int, weights:list):
    '''
    Select num_shots samples from their svm coefficient info
        with the strategy of maximum absolute value of coef.
    Args:
        svcoef:List[dict]. Key = "index" denotes te item id.
            Key = "coef" denotes the coefficient value.
        num_shots: the number of representative samples to be selected
    Return:
        select:List[Tuple]. The first element of each tuple
            denotes the item id. The second element of each
            tuple denotes the coefficient value.
    '''
    svcoef.sort(key=lambda x: x["coef"])
    select = [(elem["index"], elem["coef"]) for elem in svcoef[:num_shots]]
    return select

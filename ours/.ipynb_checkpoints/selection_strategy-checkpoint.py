# -*- coding: utf-8 -*-

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

def sv_select_equal_max_abs(svcoef:list, num_shots:int):
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

def sv_select_min(svcoef:list, num_shots:int):
    '''
    Select num_shots samples from their coefficient info
        with the strategy of minimum value of coef.
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

def sv_select_balanced_min(svcoef:list, num_shots:int):
    '''
    Select num_shots samples from their coefficient info
        with the strategy of balanced minimum value of coef.
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
    svcoef_label = {}
    select = []
    for elem in svcoef:
        svcoef_label[elem['label']] = svcoef_label.get(elem['label'], [])
        svcoef_label[elem['label']].append(elem)
    for clabel in svcoef_label.keys():
        this_num_shots = round(len(svcoef_label[clabel]) / len(svcoef) * num_shots)
        select += [(elem["index"], elem["coef"]) for elem in svcoef_label[clabel][:this_num_shots]]
    # select = [(elem["index"], elem["coef"]) for elem in svcoef[:num_shots]]
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
    
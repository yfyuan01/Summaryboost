# -*- coding: utf-8 -*-
# Author: renjun.hrj
# 2024-06-28

import pickle

def main():
    filepath = "../datasets_serialized/bank/cv0/summaryboost_bank_cv0test_128.pkl"
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    print(type(data))
    for k, v in data.items():
        print(k, v)


if __name__ == "__main__":
    main()
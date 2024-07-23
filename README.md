# Data
The data is stored in the datasets_serialized/ folder. It contains serialized data in the .jsonl. The records are generated using the dataset_all.py in the summaryboost/ folder.
# Code
The code is stored in the summaryboost/ folder. The .json and .txt files are used to construct the prompts in our experiments. main.py is the algorithm file.
# Yifei Update 07/23
Main updates: add the iterative error rule summarization mechanism, also add resampling.

The main iteration file is stored in ours/iteration_ours.py.

Also update the rules.py by saving the scores and entropy together. The old version is stored in rules_old.py

For selection_strategy.py, add the resampling function.

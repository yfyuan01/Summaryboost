export data=all
export model=mistral-7b
export num_example=128
export optimization=xgboost
python position-bias.py --dataset ${data} --cv -1 --num_shots 16 --num_examples ${num_example} --optimization ${optimization} --llm ${model}
python evaluation.py --dataset ${data} --cv -1 --num_shots 16 --num_examples ${num_example} --llm ${model} --method ${optimization} --additional_rules True --shuffled True

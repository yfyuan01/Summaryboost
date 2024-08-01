export model=mistral-7b
python original.py --portion 32 --category all --model ${model} --shot 2
python original.py --portion 32 --category all --model ${model} --shot 4
python original.py --portion 32 --category all --model ${model} --shot 8
python original.py --portion 32 --category all --model ${model} --shot 32

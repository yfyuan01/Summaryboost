export model=mistral-7b
python main_renjun.py --portion 32 --category all --model ${model} --shot 2
python main_renjun.py --portion 32 --category all --model ${model} --shot 4
python main_renjun.py --portion 32 --category all --model ${model} --shot 8
python main_renjun.py --portion 32 --category all --model ${model} --shot 32

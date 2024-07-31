export model=gpt-3.5-turbo
python original.py --portion 128 --category bank --model ${model} --shuffled True
python original.py --portion 128 --category blood --model ${model} --shuffled True
python original.py --portion 128 --category calhousing --model ${model} --shuffled True
python original.py --portion 128 --category car --model ${model} --shuffled True
python original.py --portion 128 --category creditg --model ${model} --shuffled True
python original.py --portion 128 --category diabetes --model ${model} --shuffled True
python original.py --portion 128 --category heart --model ${model} --shuffled True
python original.py --portion 128 --category income --model ${model} --shuffled True
python original.py --portion 128 --category jungle --model ${model} --shuffled True

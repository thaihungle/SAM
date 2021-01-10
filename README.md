# SAM & STM
source code for Self-attentive Associative Memory  
arXiv version: https://arxiv.org/abs/2002.03519  
code reference for NTM tasks: https://github.com/thaihungle/NSM  
code reference for NFarthest: https://github.com/L0SG/relational-rnn-pytorch    
code reference for associative retrieval: https://github.com/jiamings/fast-weights  
code reference for babi: https://github.com/APodolskiy/TPR-RNN-Torch

# Setup  
torch 1.0.0 or 1.0.1
```
mkdir logs
mkdir saved_models
```

# Vector tasks
run command examples for Copy
``` 
LSTM baseline: python run_toys.py -task_json=./tasks/copy.json -model_name=lstm -mode=train
STM: python run_toys.py -task_json=./tasks/copy.json -model_name=stm -mode=train
```
for Priority Sort 
``` 
LSTM baseline: python run_toys.py -task_json=./tasks/prioritysort.json -model_name=lstm -mode=train
STM: python run_toys.py -task_json=./tasks/prioritysort.json -model_name=stm -mode=train
```
for RAR 
``` 
LSTM baseline: python run_toys.py -task_json=./tasks/rar.json -model_name=lstm -mode=train
STM: python run_toys.py -task_json=./tasks/rar.json -model_name=stm -mode=train
```
for NFarthest 
``` 
LSTM baseline: python run_toys.py -task_json=./tasks/nfar.json -model_name=lstm -mode=train
STM: python run_toys.py -task_json=./tasks/nfar.json -model_name=stm -mode=train
```

# Associative retrieval task
generate data  
```
cd datasets/number_arecall
python number_arecall.py
```
STM training  
``` 
python run_nar.py -task_json=./tasks/nar.json -mode=train
```

# Babi task
training
```
python run_all_babi.py 
```
testing
```
python run_all_babi.py --eval-test
```

# RL task
training
```
python run_rl.py --skip_rate 32
```
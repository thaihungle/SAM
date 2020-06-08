import json
from tqdm import tqdm
import numpy as np

import os
import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value
from datasets import NARDataset
import random
from args import get_parser


def compute(data, model, bs):
    input, target = data[0], data[1]
    out = torch.zeros([1, target.size(0), dataset.out_dim])
    if torch.cuda.is_available():
        input, target, out = input.cuda(), target.cuda(), out.cuda()

    input =  model.emb(input)
    out, _ = model(input)
    out = torch.reshape(out, [-1, out_dim])
    target = torch.reshape(target, [-1, ])

    return out, target

args = get_parser().parse_args()

task_params = json.load(open(args.task_json))
args.task_name = task_params['task']
if 'iter' in task_params:
    args.num_iters = task_params['iter']
log_dir = os.path.join(args.log_dir,args.task_name)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
log_dir = os.path.join(log_dir, args.model_name)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)



save_dir = os.path.join(args.save_dir,args.task_name+args.model_name)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

save_dir = os.path.join(save_dir,"{}.pt".format(args.model_name))



dataset = NARDataset(task_params)

in_dim = dataset.in_dim
out_dim = dataset.out_dim

from baselines.sam.stm_basic import STM

model = STM(100, out_dim,
                num_slot=task_params['num_slot'],
                slot_size=task_params['slot_size'],
                rel_size=task_params['rel_size'],
                rd=("rd" not in task_params),
                init_alphas = [None, None, None])

model.emb = nn.Embedding(dataset.in_dim, 100)

print(model)
if torch.cuda.is_available():
    model.cuda()


print("====num params=====")

print(model.calculate_num_params())

print("========")

if args.resume is not None:
    print(f"resume model ... {args.resume}")
    if args.resume == "":
        model.load_state_dict(torch.load(save_dir))
    else:
        model.load_state_dict(torch.load(args.resume))

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.RMSprop(model.parameters(),
                          lr=1e-4,
                          momentum=0.9)
# optimizer = optim.Adam(model.parameters(),
#                           lr=1e-3)

cur_dir = os.getcwd()

if args.mode=="train":
    model.train()
    num_iter = args.num_iters
    print("===training===")
    configure(log_dir)
elif args.mode=="test":
    model.eval()
    num_iter = dataset.ar_data.test._num_examples//args.batch_size
    print("===testing===")
    model.load_state_dict(torch.load(save_dir))
    print(f"load weight {save_dir}")
    torch.manual_seed(5111)
    torch.cuda.manual_seed(1111)
    np.random.seed(1111)
    random.seed(1111)
# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
loss_pls = []

best_acc = 0
train_acc = []
acc = []
epoch=0
for iter in tqdm(range(num_iter)):
    optimizer.zero_grad()
    if args.mode == "train":
        for p in model.parameters():
            p.requires_grad = True
        model.train()
        data = dataset.get_sample_wlen(bs=args.batch_size, type="train")
    elif args.mode == "test":
        data = dataset.get_sample_wlen(bs=args.batch_size, type="test")


    out, target = compute(data, model, args.batch_size)

    train_acc.append(torch.mean((torch.argmax(out, dim=-1) == target).float()).item())




    loss = criterion(out, target)


    losses.append(loss.item())

    if args.mode == "train":
        loss.backward()
        # clips gradient in the range [-10,10]. Again there is a slight but
        # insignificant deviation from the paper where they are clipped to (-10,10)
        nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
        optimizer.step()



        # ---logging---
        if iter % args.freq_val == 0 or  iter%(dataset.ar_data.train._num_examples//args.batch_size)==0:
            print('Iteration: %d\tLoss: %.2f %.2f' %
                  (iter, np.mean(losses), np.mean(train_acc)))
            mloss = np.mean(losses)
            len_data = dataset.ar_data.val._num_examples

            if  iter%(dataset.ar_data.train._num_examples//args.batch_size)==0:
                print(f"validate epoch {epoch}")
                epoch+=1

                for p in model.parameters():
                    p.requires_grad = False
                model.eval()
                counter = 0
                while counter<len_data:
                    counter2 = counter+args.batch_size
                    if counter2>len_data:
                        counter2=len_data
                    bs = counter2-counter
                    data = dataset.get_sample_wlen(bs=[counter,counter2], type="valid")
                    out, target = compute(data, model, bs)
                    acc.append(torch.mean((torch.argmax(out, dim=-1) == target).float()).item())
                    counter+=args.batch_size
                cur_acc = np.mean(acc)
                if cur_acc>best_acc:
                    # ---saving the model---
                    print(f"save the best model!!! with acc {cur_acc}")
                    torch.save(model.state_dict(), save_dir)
                    best_acc = cur_acc
                acc = []
                log_value('val_acc', cur_acc, iter//(dataset.ar_data.train._num_examples//args.batch_size))
            log_value('train_loss', mloss, iter)
            log_value('train_acc', np.mean(train_acc), iter)

            losses = []
            train_acc=[]
            loss_pls = []


if args.mode=="test":
    print('test_loss', np.mean(losses))
    print(f'bacc {np.mean(train_acc)}')
import json
from tqdm import tqdm
import numpy as np
import random
import os
import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value
import torch.nn.functional as F

from datasets import NFarDataset, CopyDataset, PrioritySortDataset, RARDataset


from args import get_parser



args = get_parser().parse_args()



# ----------------------------------------------------------------------------
# -- initialize datasets, model, criterion and optimizer
# ----------------------------------------------------------------------------


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



if "nfar" in args.task_name:
    dataset = NFarDataset(task_params)
elif "copy" in args.task_name:
    dataset = CopyDataset(task_params)
elif "prioritysort" in args.task_name:
    dataset = PrioritySortDataset(task_params)
elif "rar" in args.task_name:
    dataset = RARDataset(task_params)

in_dim = dataset.in_dim
out_dim = dataset.out_dim






if 'lstm' in args.model_name:
    from baselines.nvm.lstm_baseline import LSTMBaseline
    hidden_dim = task_params['controller_size']*2
    model = LSTMBaseline(in_dim, hidden_dim, out_dim, 1)
elif 's2s_att' in args.model_name:
    from baselines.nvm.lstm_att_baseline import AttnEncoderDecoder
    hidden_dim = task_params['controller_size']*2
    model = AttnEncoderDecoder(in_dim, out_dim, hidden_dim, max_att_len=100)
elif 'stm' in args.model_name:
    from baselines.sam.stm_basic import STM
    if "alphas" in task_params:
        alphas = task_params["alphas"]
    else:
        alphas = [None, None, None]

    model = STM(in_dim, out_dim,
                num_slot=task_params['num_slot'],
                slot_size=task_params['slot_size'],
                rel_size=task_params['rel_size'],
                rd=("rd" not in task_params),
                init_alphas = alphas)
else:
    from baselines.nvm.ntm_warper import EncapsulatedNTM
    model = EncapsulatedNTM(
        num_inputs=in_dim,
        num_outputs=out_dim,
        controller_size=task_params['controller_size']*2,
        controller_layers =1,
        num_heads = task_params['num_heads'],
        N=task_params['memory_units'],
        M=task_params['memory_unit_size'])

print(model)
if torch.cuda.is_available():
    model.cuda()

print("====num params=====")

print(model.calculate_num_params())

print("========")

if "nfar" not in args.task_name:
    criterion = nn.BCELoss()
else:
    criterion = nn.CrossEntropyLoss()
# As the learning rate is task specific, the argument can be moved to json file
optimizer = optim.RMSprop(model.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          momentum=args.momentum)

if 'nfar' in args.task_name:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)



cur_dir = os.getcwd()


# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
errors = []
rel_errors = []
loss_pls = []

best_loss = 10000

print(args)

if args.mode=="train":
    model.train()
    num_iter = args.num_iters
    print("===training===")
    configure(log_dir)
elif args.mode=="test":
    model.eval()
    num_iter = args.num_eval
    print("===testing===")
    model.load_state_dict(torch.load(save_dir))
    print(f"load weight {save_dir}")
    torch.manual_seed(5111)
    torch.cuda.manual_seed(1111)
    np.random.seed(1111)
    random.seed(1111)

for iter in tqdm(range(num_iter)):
    optimizer.zero_grad()
    model.init_sequence(batch_size=args.batch_size)
    if "nfar" in args.task_name:
        data = dataset.get_sample_wlen(bs=args.batch_size)
    elif "copy" in args.task_name:
        random_length = np.random.randint(task_params['min_seq_len'],
                                          task_params['max_seq_len'] + 1)

        data = dataset.get_sample_wlen(random_length, bs=args.batch_size)
    elif "prioritysort"  in args.task_name:
        data = dataset.get_sample_wlen(bs=args.batch_size)
    elif "rar" in args.task_name:
        num_item = torch.randint(
            task_params['min_item'], task_params['max_item'], (1,), dtype=torch.long).item()
        data = dataset.get_sample_wlen(task_params['seq_len'], num_item, bs=args.batch_size)

    if torch.cuda.is_available():
        input, target = data['input'].cuda(), data['target'].cuda()
        out = torch.zeros(target.size()).cuda()
    else:
        input, target = data['input'], data['target']
        out = torch.zeros(target.size())


    # -------------------------------------------------------------------------
    # loop for other tasks
    # -------------------------------------------------------------------------
    if "lstm" in args.model_name or "stm" in args.model_name \
             or "ntm" in args.model_name:
        for i in range(input.size()[0]):
            in_data = input[i]
            sout, _ = model(in_data)

        if "nfar" not in args.task_name:
            if torch.cuda.is_available():
                in_data = torch.zeros(input.size()).cuda()
            else:
                in_data = torch.zeros(input.size())


            for i in range(target.size()[0]):
                sout, _ = model(in_data[-1])
                out[i] = F.sigmoid(sout)
        else:
            out[-1] = sout
    elif "s2s_att" in args.model_name:
        out, _, = model(input, target_length=target.shape[0])
        if "nfar" not in args.task_name:
            out = F.sigmoid(out)




        # -------------------------------------------------------------------------
    if "nfar" in args.task_name:
        loss = criterion(torch.reshape(out, [-1, dataset.out_dim]),
                  torch.argmax(torch.reshape(target, [-1, dataset.out_dim]), -1))
        loss = torch.mean(loss)
    else:
        loss = criterion(out, target)

    losses.append(loss.item())

    if args.mode=="train":
        loss.backward()
        if args.clip_grad > 0:
            nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

    binary_output = out.clone()
    if torch.cuda.is_available():
        binary_output = binary_output.detach().cpu().apply_(lambda x: 0 if x < 0.5 else 1).cuda()
    else:
        binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)
    if "nfar" not in args.task_name:
        # sequence prediction error is calculted in bits per sequence
        error = torch.sum(torch.abs(binary_output - target))/args.batch_size
        # if 'rar' in args.task_name:
        #     abs_err = 0.0
        #     for b in range(args.batch_size):
        #         abs_err+=(torch.sum(binary_output[:,b,:]!=target[:,b,:])>0)
        #     abs_err=abs_err.float()/args.batch_size
        #     print("sss")
        #     print(abs_err)
    else:
        error = torch.sum(torch.argmax(out, dim=-1) != torch.argmax(target, dim=-1)).float()/(target.shape[1]*target.shape[0])


    errors.append(error.item())
    rel_errors.append((error*args.batch_size/torch.sum(target)).item())
    # ---logging---
    if args.mode=="train" and iter % args.freq_val == 0:
        print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
              (iter, np.mean(losses), np.mean(errors)))
        mloss = np.mean(losses)

        if mloss<best_loss:
            # ---saving the model---
            torch.save(model.state_dict(), save_dir)
            best_loss = mloss
        log_value('train_loss', mloss, iter)
        log_value('bit_error_per_sequence', np.mean(errors), iter)

        losses = []
        errors = []
        loss_pls = []



if args.mode=="test":
    print('test_loss', np.mean(losses))
    print(f'bit_error_per_sequence {np.mean(errors)} -->{np.mean(rel_errors)} ')
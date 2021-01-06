import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import random

def prepare_sample_batch(samples, end_token, out_dim, max_len = 500, random_mode=True):
    max_seq_len=0
    max_out_len=0
    for index in range(len(samples)):
        # print(sample[index]['inputs'])
        # print(sample[index]['outputs'])
        max_seq_len=max(len(samples[index]['inputs']),max_seq_len)
        max_out_len=max(len(samples[index]['outputs']),max_out_len)

    max_seq_len = min(max_seq_len, max_len)
    max_out_len = min(max_out_len, max_len)+1
    total_seq_len = max_seq_len+1

    input_vecs=[]
    output_vecs=[]
    for index in range(len(samples)):
        # print('\n{}'.format(index))
        ins=samples[index]['inputs']
        outs=samples[index]['outputs']
        input_vec = np.zeros([total_seq_len, 2+out_dim])
        output_vec = np.zeros(max_out_len, dtype=np.long)
        # print(samples[index]['inputs'])
        # print(target_code)
        # print(samples[index]['outputs'])
        offset = max_seq_len - min(len(ins), max_seq_len)
        if random_mode:
            labels = np.random.choice(max_seq_len, max_seq_len, replace=False)
        for iii, token in enumerate(ins):
            if iii==max_seq_len:
                break
            input_vec[iii+offset][0] = float(token[0])
            input_vec[iii+offset][1] = float(token[1])

            if random_mode:
                pos = labels[iii]+1
            else:
                pos = iii+1
            input_vec[iii + offset][2+pos] = 1.0
        input_vec[iii + offset][2] = 1.0
        # for i in range(offset):
        #     input_vec[i][2] = 1
        # print(outs)
        for iii, token in enumerate(outs):
            if iii==max_out_len:
                break
            if random_mode:
                out_value = labels[int(token)-1]+1
            else:
                out_value = int(token)
            output_vec[iii] = out_value
        output_vec[iii]=end_token
        # print(input_vec)
        # print(output_vec)
        # print(mask)
        # print('====')
        # input_vec = [onehot(code, word_space_size) for code in input_vec]
        # output_vec = [onehot(code, word_space_size) for code in output_vec]
        input_vecs.append(input_vec)
        output_vecs.append(output_vec)


    input_vecs = np.transpose(np.asarray(input_vecs),[1,0,2])
    output_vecs = np.transpose(np.asarray(output_vecs),[1,0])
    # raise False
    return torch.tensor(input_vecs, dtype=torch.float), \
           torch.tensor(output_vecs, dtype=torch.long), \
           total_seq_len, max_out_len

class TSPDataset(Dataset):


    def __init__(self, task_params,  curriculum=False, mode='train'):
        """Initialize a dataset instance for copy task.

        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to copy task.
        """
        self.task_params = task_params
        self.cur_index = 0

        if mode=="train":
            self.train_samples = self.read_file(task_params['data_dir_train'], same_len=True)
            for k,v in self.train_samples.items():
                print(f"num train of len {k} {len(v)}")
            if curriculum:
                self.train_samples.sort(key=lambda x: len(x["inputs"]))
        if mode=="test":
            self.test_samples = self.read_file(task_params['data_dir_test'])
            # for k,v in self.test_samples.items():
            print(f"num test of len {len(self.test_samples)}")
        self.out_dim = task_params["N_max"]+2
        self.in_dim = 2 + self.out_dim
        self.cur_index = -1

    """
    assume no batch, no pad, no random label
    """
    def get_path_len_naive(self, inputs, pred):
        total_dis = 0
        num_p = len(inputs)-1
        for pi, pv in enumerate(pred):
            if pi+1<=num_p:
                if pred[pi + 1] == self.out_dim - 1:
                    nexp = inputs[0][:2]
                    curp = inputs[pv-1][:2]
                    dis = torch.norm(curp - nexp, 2)
                    total_dis += dis
                elif pred[pi+1]!=self.out_dim-1:
                    ni = pred[pi+1]-1
                    if ni<0 or ni>=num_p:
                        ni=0
                    ci = pv-1
                    if ci<0 or ci>=num_p:
                        ci=num_p-1
                    nexp = inputs[ni][:2]
                    curp = inputs[ci][:2]
                    dis = torch.norm(curp-nexp, 2)
                    total_dis+=dis
        return total_dis





    def read_file(self, filepaths, same_len=False):
        all_data_blen =[]
        if same_len:
            all_data_blen = {}
        for filepath in filepaths:
            print(filepath)
            with open(filepath) as fp:
                for line in tqdm(fp):
                    xs = []
                    ys = []
                    all_items = line.strip().split()
                    after_output = False
                    i = 0
                    while i < len(all_items):
                        if not after_output:
                            if  all_items[i] == "output":
                                after_output = True
                            else:
                                xs.append([all_items[i], all_items[i+1]])
                                i+=1
                        else:
                            ys.append(all_items[i])
                        i+=1
                    # if len(xs)==10:
                    #     plot_geo.plot_points(xs, ys)
                    if len(xs)<=self.task_params["N_max"]:
                        if same_len:
                            if len(xs) not  in all_data_blen:
                                all_data_blen[len(xs)]=[]
                            all_data_blen[len(xs)].append({"inputs":xs,"outputs":ys})
                        else:
                            all_data_blen.append({"inputs":xs,"outputs":ys})
        return all_data_blen




    # def __len__(self):
    #     # sequences are generated randomly so this does not matter
    #     # set a sufficiently large size for data loader to sample mini-batches
    #     return 65536
    #
    # def __getitem__(self, idx):
    #
    #


    def get_train_sample_wlen(self, bs=1):

        if self.cur_index<0:
            chosen_key = random.choice(list(self.train_samples.keys()))
            samples = np.random.choice(self.train_samples[chosen_key], bs)
            data = prepare_sample_batch(samples, end_token=self.out_dim-1, out_dim=self.out_dim)
            return data
        else:
            find = self.cur_index
            tind = self.cur_index+bs
            if tind>len(self.train_samples):
                tind = len(self.train_samples)
                find = tind-bs
                self.cur_index=0
            else:
                self.cur_index+=bs
            samples = self.train_samples[find:tind]
            data = prepare_sample_batch(samples, end_token=self.out_dim-1, out_dim=self.out_dim)
            return data

    def get_test_sample_wlen(self, bs=1):
        if self.cur_index<0:
            self.cur_index=0
        find = self.cur_index
        tind = self.cur_index + bs
        if tind > len(self.test_samples):
            tind = len(self.test_samples)
            find = tind - bs
            self.cur_index = 0
        else:
            self.cur_index += bs
        samples = self.test_samples[find:tind]
        data = prepare_sample_batch(samples, end_token=self.out_dim - 1, out_dim=self.out_dim, random_mode=False)
        return data
import torch
from torch.utils.data import Dataset
from torch.distributions.uniform import Uniform
from torch.distributions.binomial import Binomial


class PrioritySortDataset(Dataset):
    """A Dataset class to generate random examples for priority sort task.

    In the input sequence, each vector is generated randomly along with a
    scalar priority rating. The priority is drawn uniformly from the range
    [-1,1) and is provided on a separate input channel.

    The target contains the binary vectors sorted according to their priorities
    """

    def __init__(self, task_params):
        """ Initialize a dataset instance for the priority sort task.

        Arguments
        ---------
        task_params : dict
                A dict containing parameters relevant to priority sort task.
        """
        self.seq_width = task_params["seq_width"]
        self.input_seq_len = task_params["input_seq_len"]
        self.target_seq_len = task_params["target_seq_len"]
        self.in_dim = task_params['seq_width'] + 2
        self.out_dim = task_params['seq_width']

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        prob = 0.5 * torch.ones([self.input_seq_len,
                                 self.seq_width], dtype=torch.float64)
        seq = Binomial(1, prob).sample()
        # Extra input channel for providing priority value
        input_seq = torch.zeros([self.input_seq_len, self.in_dim])
        input_seq[:self.input_seq_len, :self.seq_width] = seq

        # torch's Uniform function draws samples from the half-open interval
        # [low, high) but in the paper the priorities are drawn from [-1,1].
        # This minor difference is being ignored here as supposedly it doesn't
        # affects the task.
        priority = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
        for i in range(self.input_seq_len):
            input_seq[i, self.seq_width] = priority.sample()

        sorted, ind = torch.sort(input_seq[:, self.seq_width], 0, descending=True)
        sorted = input_seq[ind]
        target_seq = sorted[:self.target_seq_len, :self.seq_width]

        return {'input': input_seq, 'target': target_seq}

    def get_sample_wlen(self,  bs=1):
        # idx only acts as a counter while generating batches.
        prob = 0.5 * torch.ones([self.input_seq_len, bs,
                                 self.seq_width], dtype=torch.float64)
        seq = Binomial(1, prob).sample()
        # Extra input channel for providing priority value
        input_seq = torch.zeros([self.input_seq_len, bs, self.in_dim])
        input_seq[:self.input_seq_len,:,:self.seq_width] = seq

        # torch's Uniform function draws samples from the half-open interval
        # [low, high) but in the paper the priorities are drawn from [-1,1].
        # This minor difference is being ignored here as supposedly it doesn't
        # affects the task.
        priority = Uniform(torch.tensor([-1.0]*bs), torch.tensor([1.0]*bs))
        for i in range(self.input_seq_len):
            input_seq[i,:,self.seq_width] = priority.sample()

        target_seq = []
        for j in range(bs):
            sorted, ind = torch.sort(input_seq[:,j,self.seq_width], 0, descending=True)
            sorted = input_seq[ind,j]
            target_seq.append(sorted[:self.target_seq_len,:self.seq_width].unsqueeze(1))
        target_seq = torch.cat(target_seq, 1)
        return {'input': input_seq, 'target': target_seq}

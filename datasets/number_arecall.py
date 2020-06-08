import pickle
from torch.utils.data import Dataset
import torch
import numpy  as np

class NARDataset(Dataset):
    """A Dataset class to generate random examples for the copy task. Each
    sequence has a random length between `min_seq_len` and `max_seq_len`.
    Each vector in the sequence has a fixed length of `seq_width`. The vectors
    are bounded by start and end delimiter flags.

    To account for the delimiter flags, the input sequence length as well
    width is two more than the target sequence.
    """

    def __init__(self, task_params):
        """Initialize a dataset instance for copy task.

        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to copy task.
        """
        self.ar_data = read_data(task_params["data_dir"])
        self.in_dim = 26 + 10 + 1
        self.out_dim = 10

        print(f"num train {self.ar_data.train._num_examples}")
        print(f"num valid {self.ar_data.val._num_examples}")
        print(f"num test {self.ar_data.test._num_examples}")


    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        pass

    def get_sample_wlen(self, bs=1, type="train"):
        if type=="train":
            bx, by = self.ar_data.train.next_batch(batch_size=bs)
        elif type =="test":
            bx, by = self.ar_data.test.next_batch(batch_size=bs)
        else:
            bx, by = self.ar_data.val.next_batch(batch_size=bs)


        return torch.tensor(bx).permute(1,0).long(), torch.tensor(by).long()

class Dataset(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._epoch_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self.x.shape[0]
        self.perm = np.random.permutation(np.arange(self._num_examples))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        if isinstance(batch_size, int):
            assert batch_size <= self._num_examples
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch >= self.num_examples:
                self._epoch_completed += 1
                np.random.shuffle(self.perm)
                start = 0
                self._index_in_epoch = batch_size
            end = self._index_in_epoch
            return self._x[self.perm[start:end]], self._y[self.perm[start:end]]
        else:
            start, end = batch_size[0], batch_size[1]
            return self._x[self.perm[start:end]], self._y[self.perm[start:end]]

import collections

Datasets = collections.namedtuple('Datasets', ['train', 'val', 'test'])


def read_data(data_path='./number_arecall/associative-retrieval.pkl'):
    with open(data_path, 'rb') as f:
        d = pickle.load(f)
    x_train = d['x_train']
    x_val = d['x_val']
    x_test = d['x_test']
    y_train = d['y_train']
    y_val = d['y_val']
    y_test = d['y_test']
    train = Dataset(x_train, y_train)
    test = Dataset(x_test, y_test)
    val = Dataset(x_val, y_val)
    return Datasets(train=train, val=val, test=test)
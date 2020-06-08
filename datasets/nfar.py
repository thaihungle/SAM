import numpy as np
import torch
from torch.utils.data import Dataset



####################
# Generate data
####################


def one_hot_encode(array, num_dims=8):
    one_hot = np.zeros((len(array), num_dims))
    for i in range(len(array)):
        one_hot[i, array[i]] = 1
    return one_hot


def get_example(num_vectors, num_dims):
    input_size = num_dims + num_vectors * 3
    n = np.random.choice(num_vectors, 1)  # nth farthest from target vector
    labels = np.random.choice(num_vectors, num_vectors, replace=False)
    m_index = np.random.choice(num_vectors, 1)  # m comes after the m_index-th vector
    m = labels[m_index]

    # Vectors sampled from U(-1,1)
    vectors = np.random.rand(num_vectors, num_dims) * 2 - 1
    target_vector = vectors[m_index]
    dist_from_target = np.linalg.norm(vectors - target_vector, axis=1)
    X_single = np.zeros((num_vectors, input_size))
    X_single[:, :num_dims] = vectors
    labels_onehot = one_hot_encode(labels, num_dims=num_vectors)
    X_single[:, num_dims:num_dims + num_vectors] = labels_onehot
    nm_onehot = np.reshape(one_hot_encode([n, m], num_dims=num_vectors), -1)
    X_single[:, num_dims + num_vectors:] = np.tile(nm_onehot, (num_vectors, 1))
    y_single = labels[np.argsort(dist_from_target)[-(n + 1)]]

    return X_single, y_single

def get_example2(num_vectors, num_dims):
    input_size = num_dims
    m_index = np.random.choice(num_vectors, 1)  # m comes after the m_index-th vector
    prob = 0.5 * np.ones([num_vectors, num_dims])
    vectors = np.random.binomial(1, prob)
    # Vectors sampled from U(-1,1)
    # vectors = np.random.rand(num_vectors, num_dims) * 2 - 1
    target_vector = vectors[m_index]
    dist_from_target = np.linalg.norm(vectors - target_vector, axis=1)
    X_single = np.zeros((num_vectors, input_size))
    X_single[:, :num_dims] = vectors

    y_index = np.argsort(dist_from_target)[1]
    y_single = vectors[y_index]
    pad = np.zeros([num_vectors, num_dims])
    X_single = np.concatenate([X_single, target_vector, pad], axis=0)
    return X_single, y_single


def get_examples(num_examples, num_vectors, num_dims, input_size, device=0):
    X = np.zeros((num_examples, num_vectors, input_size))
    y = np.zeros((num_examples, 1, num_vectors))
    for i in range(num_examples):
        X_single, y_single = get_example(num_vectors, num_dims)
        X[i, :] = X_single
        y[i][0][y_single[0]]=1
    X = np.transpose(X, [1,0,2])
    y = np.transpose(y, [1,0,2])

    X = torch.Tensor(X).to(device)
    y = torch.LongTensor(y).to(device)

    return X, y


class NFarDataset(Dataset):
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
        self.num_dims = task_params['num_dims']
        self.num_vectors = task_params['num_vectors']
        self.in_dim = self.num_dims + self.num_vectors * 3

        self.out_dim = self.num_vectors

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        input_seq, target_seq = get_examples(1, self.num_vectors, self.num_dims, self.in_dim)
        return {'input': input_seq[:,0,:], 'target': target_seq[:,0,:]}

    def get_sample_wlen(self, bs=1):
        input_seq, target_seq = get_examples(bs, self.num_vectors, self.num_dims, self.in_dim)
        return {'input': input_seq, 'target': target_seq}


"""NTM Read and Write Heads."""
from baselines.nvm.util import *

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard

def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class NTMHeadBase(nn.Module):
    """An NTM Read/Write Head."""

    def __init__(self, memory, controller_size):
        """Initilize the read/write head.
        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super(NTMHeadBase, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size
        self.cs = []
        self.program_weights=[]
        self.query_keys=[]
        self.query_strengths=[]
        self.data_weights=[]
        self.att_mode = "kv"


    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, β, g, s, γ, w_prev):
        # Handle Activations
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)

        w = self.memory.address(k, β, g, s, γ, w_prev)

        return w

    def read_mem(self, memory, read_weights, key_size):
        return torch.bmm(read_weights, memory[:,:,key_size:])

    def content_weightings(self, memory, keys, strengths, key_size, program_mask=None):
        if key_size>0:
            if self.att_mode=="kv":
                d = θ(F.tanh(memory[:,:,:key_size]), F.tanh(keys[:,:,:key_size]))
            else:
                d = keys
        else:
            d = θ(memory, keys)
        # print(memory[:,:,:key_size])
        d = σ(d * strengths.unsqueeze(2), 2)
        if program_mask is not None:
        #     # d = torch.abs(d*program_mask)
        #     # print(d)
        #     # d2 = torch.zeros(d.shape).cuda()
        #     # _, di = d.max(-1)
        #     # d2[:,:,di]=1
        #     # d=d2
            d = gumbel_softmax(d, 10)
        #     # print(d)
        return d

class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size, program_size=2, pkey_dim=2):
        super(NTMReadHead, self).__init__(memory, controller_size)
        self.program_size = program_size
        self.program_mask = None
        self.pkey_dim = pkey_dim
        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = [self.M, 1, 1, 3, 1]
        self.layernorm = nn.GroupNorm(1, sum(self.read_lengths))

        if self.program_size>0:
            self.program_key = nn.Linear(controller_size, self.pkey_dim)
            self.program_strength = nn.Linear(controller_size, 1)

            self.instruction_weight = nn.Parameter(torch.zeros(self.program_size,
                                                               self.pkey_dim +
                                                               (self.controller_size+1)*sum(self.read_lengths),
                                                               requires_grad=True))
        else:
            self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))

        self.reset_parameters()

    def create_new_state(self, batch_size):
        self.cs = []
        self.program_weights = []
        self.query_keys = []
        self.query_strengths=[]
        self.data_weights = []
        # The state holds the previous time step address weightings
        if torch.cuda.is_available():
            return torch.zeros(batch_size, self.N).cuda()
        else:
            return torch.zeros(batch_size, self.N)


    def reset_parameters(self):
        # Initialize the linear layers
        if self.program_size>0:
            nn.init.xavier_uniform_(self.instruction_weight, gain=1.4)
            nn.init.xavier_uniform_(self.program_key.weight, gain=1.4)
            nn.init.normal_(self.program_key.bias, std=0.01)
            nn.init.xavier_uniform_(self.program_strength.weight, gain=1.4)
            nn.init.normal_(self.program_strength.bias, std=0.01)
        else:
            nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
            nn.init.normal_(self.fc_read.bias, std=0.01)


    def is_read_head(self):
        return True

    def forward(self, embeddings, w_prev):
        """NTMReadHead forward function.
        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        self.cs.append(embeddings[0])
        if self.program_size>0:
            # if len(self.query_keys)>0:
            #     read_keys = self.query_keys[0]
            #     read_strengths = self.query_strengths[0]
            # else:
            read_keys = self.program_key(embeddings)
            read_strengths = F.softplus(self.program_strength(embeddings))
            content_weights = self.content_weightings(self.instruction_weight.unsqueeze(0).repeat(read_keys.shape[0],1,  1),
                                                      read_keys.unsqueeze(1),
                                                      read_strengths, self.pkey_dim, self.program_mask)
            instruction = self.read_mem(self.instruction_weight.unsqueeze(0).repeat(read_keys.shape[0],1, 1),
                                        content_weights, self.pkey_dim)
            i_w = instruction[:,:,:self.controller_size*sum(self.read_lengths)].view(-1, self.controller_size, sum(self.read_lengths))
            i_b = instruction[:,:,self.controller_size*sum(self.read_lengths):].view(-1, 1,  sum(self.read_lengths))

            o = (torch.matmul(embeddings.unsqueeze(1), i_w)+i_b).squeeze(1)
            self.program_weights.append(content_weights)
            # print(content_weights)
            self.query_keys.append(read_keys)
            self.query_strengths.append(read_strengths)
        else:
            o = self.fc_read(embeddings)
        # o = self.layernorm(o)
        k, β, g, s, γ = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self._address_memory(k, β, g, s, γ, w_prev)
        r = self.memory.read(w)
        self.data_weights.append(w)
        return r, w


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size, program_size=2, pkey_dim=2):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.program_size = program_size
        self.program_mask = None
        self.pkey_dim = pkey_dim
        self.layernorm = nn.GroupNorm(1, sum(self.write_lengths))

        if self.program_size>0:
            self.program_key = nn.Linear(controller_size, self.pkey_dim)
            self.program_strength = nn.Linear(controller_size, 1)

            self.instruction_weight = nn.Parameter(torch.zeros(self.program_size,
                                                               self.pkey_dim +
                                                               (self.controller_size+1)*sum(self.write_lengths),
                                                               requires_grad=True))
        else:
            self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        self.cs = []
        self.program_weights = []
        self.query_keys = []
        self.query_strengths = []
        self.data_weights = []
        if torch.cuda.is_available():
            return torch.zeros(batch_size, self.N).cuda()
        else:
            return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initialize the linear layers
        if self.program_size>0:
            nn.init.xavier_uniform_(self.instruction_weight, gain=1.4)
            nn.init.xavier_uniform_(self.program_key.weight, gain=1.4)
            nn.init.normal_(self.program_key.bias, std=0.01)
            nn.init.xavier_uniform_(self.program_strength.weight, gain=1.4)
            nn.init.normal_(self.program_strength.bias, std=0.01)
        else:
            nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
            nn.init.normal_(self.fc_write.bias, std=0.01)


    def is_read_head(self):
        return False

    def forward(self, embeddings, w_prev):
        """NTMWriteHead forward function.
        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        self.cs.append(embeddings[0])
        if self.program_size>0:
            # if len(self.query_keys)>0:
            #     read_keys = self.query_keys[0]
            #     read_strengths = self.query_strengths[0]
            # else:
            read_keys = self.program_key(embeddings)
            read_strengths = F.softplus(self.program_strength(embeddings))
            content_weights = self.content_weightings(self.instruction_weight.unsqueeze(0).repeat(read_keys.shape[0],1,  1),
                                                      read_keys.unsqueeze(1),
                                                      read_strengths, self.pkey_dim, self.program_mask)
            instruction = self.read_mem(self.instruction_weight.unsqueeze(0).repeat(read_keys.shape[0], 1, 1),
                                        content_weights, self.pkey_dim)

            i_w = instruction[:, :, :self.controller_size * sum(self.write_lengths)].view(-1, self.controller_size,
                                                                                         sum(self.write_lengths))
            i_b = instruction[:, :, self.controller_size * sum(self.write_lengths):].view(-1, 1, sum(self.write_lengths))

            o = (torch.matmul(embeddings.unsqueeze(1), i_w) + i_b).squeeze(1)
            self.program_weights.append(content_weights)
            # print(content_weights)
            self.query_keys.append(read_keys)
            self.query_strengths.append(read_strengths)
            # u, s, d = torch.svd(instruction[0])
            # print(s[0])
        else:
            o = self.fc_write(embeddings)
            # u, s, d = torch.svd(self.fc_write.weight)
            # print(s[0])

        # o = self.layernorm(o)
        k, β, g, s, γ, e, a = _split_cols(o, self.write_lengths)

        # e should be in [0, 1]
        e = F.sigmoid(e)

        # Write to memory
        w = self._address_memory(k, β, g, s, γ, w_prev)
        self.memory.write(w, e, a)
        self.data_weights.append(w)
        return w
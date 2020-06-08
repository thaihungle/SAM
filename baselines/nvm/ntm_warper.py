"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from .ntm import NTM
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .ntm_mem import NTMMemory
import numpy as np
import torch.nn.functional as F


class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M,
                 program_size=0, pkey_dim=0):
        """Initialize an EncapsulatedNTM.
        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M
        self.program_size = program_size
        self.pkey_dim = pkey_dim
        self.emb = None

        # Create the NTM components
        memory = NTMMemory(N, M)
        controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
        self.heads = nn.ModuleList([])
        for i in range(num_heads):
            self.heads += [
                NTMReadHead(memory, controller_size, self.program_size, self.pkey_dim),
                NTMWriteHead(memory, controller_size, self.program_size, self.pkey_dim)
            ]

        self.ntm = NTM(num_inputs, num_outputs, controller, memory, self.heads)
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)

    def forward(self, x=None):
        if self.emb is not None:
            x = self.emb(x.long()).squeeze(1)

        if x is None:
            if torch.cuda.is_available():
                x = torch.zeros(self.batch_size, self.num_inputs).cuda()
            else:
                x = torch.zeros(self.batch_size, self.num_inputs)
        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state

    def program_loss_pl1(self):
        ploss = 0
        count = 0
        for head in self.heads:
            for i in range(self.program_size):
                for j in range(i + 1, self.program_size):
                    ploss += F.cosine_similarity \
                        (head.instruction_weight[i, :self.pkey_dim],
                         head.instruction_weight[j, :self.pkey_dim],
                         dim=0)
                    count += 1
        return ploss / count

    def set_program_mask(self, pm):
        for head in self.heads:
            head.program_mask=pm

    def set_att_mode(self, mode="kv"):
        for head in self.heads:
            print("set att mode to: {}".format(mode))
            head.att_mode=mode

    def program_loss_pl2(self):
        ploss = 0
        count = 0
        if torch.cuda.is_available():
            I = torch.eye(self.program_size).cuda()
        else:
            I = torch.eye(self.program_size)

        for head in self.heads:
            W = head.instruction_weight[:, :self.pkey_dim]
            ploss += torch.norm(torch.matmul(W, torch.t(W))-I)
            count+=1
        return ploss / count

    def get_read_meta_info(self):
        meta={"read_program_weights":[],
              "read_query_keys":[],
              "read_program_keys":[],
              "write_program_weights": [],
              "write_query_keys": [],
              "write_program_keys": [],
              "read_data_weights":[],
              "write_data_weights":[],
              "css":[]
              }
        for head in self.heads:
            meta["css"].append(head.cs)
            if head.is_read_head():
                if self.program_size>0:
                    meta["read_program_weights"].append(head.program_weights)
                    meta["read_program_keys"].append(head.instruction_weight[:, :self.pkey_dim])
                    meta["read_query_keys"].append(head.query_keys)

                meta["read_data_weights"].append(head.data_weights)
            else:
                if self.program_size>0:
                    meta["write_program_weights"].append(head.program_weights)
                    meta["write_program_keys"].append(head.instruction_weight[:, :self.pkey_dim])
                    meta["write_query_keys"].append(head.query_keys)

                meta["write_data_weights"].append(head.data_weights)

        for k,vv in meta.items():
            for i1, v in enumerate(vv):
                if isinstance(v,list):
                    for i2, v2 in enumerate(v):
                        meta[k][i1][i2] = np.asarray(v2.detach().cpu())
                else:
                    meta[k][i1] = np.asarray(vv[i1].detach().cpu())

        return meta

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
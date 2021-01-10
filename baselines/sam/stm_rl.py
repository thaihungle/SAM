import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def op_att(q, k, v):
    qq = q.unsqueeze(2).repeat(1, 1, k.shape[1], 1)
    kk = k.unsqueeze(1).repeat(1, q.shape[1], 1, 1)
    output = torch.matmul(F.tanh(qq*kk).unsqueeze(4), v.unsqueeze(1).repeat(1, q.shape[1], 1, 1).unsqueeze(3))  # BxNXNxd_kq BxNxNxd_v --> BxNXNxd_kqxd_v
    # print(output.shape)
    output = torch.sum(output, dim=2)  # BxNxd_kqxd_v
    # print(output.shape)
    return output

def sdp_att(q,k,v):
    dot_product = torch.matmul(q, k.permute(0, 2,1))
    weights = F.softmax(dot_product, dim=-1)

    # output is [B, H, N, V]
    output = torch.matmul(weights, v)
    return output


class MLP(nn.Module):
    def __init__(self, in_dim=28*28,  out_dim=10, hid_dim=-1, layers=1):
        super(MLP, self).__init__()
        self.layers = layers
        if hid_dim<=0:
            self.layers=-1
        if self.layers<0:
            hid_dim=out_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        # linear layer (n_hidden -> hidden_2)
        if self.layers>0:
            self.fc2h = nn.ModuleList([nn.Linear(hid_dim, hid_dim)]*self.layers)
        # linear layer (n_hidden -> 10)
        if self.layers>=0:
            self.fc3 = nn.Linear(hid_dim, out_dim)


    def forward(self, x):
        o = self.fc1(x)
        if self.layers>0:
            for l in range(self.layers):
                o = self.fc2h[l](o)
        if self.layers >= 0:
            o = self.fc3(o)
        return o

class STM(nn.Module):
    def __init__(self, input_size, output_size, step = 1, num_slot=8,
                 mlp_size = 128, slot_size = 96, rel_size = 96,
                 out_att_size=64, rd=True,
                 init_alphas=[None,None,None],
                 learn_init_mem=True, mlp_hid=-1):
        super(STM, self).__init__()
        self.mlp_size = mlp_size
        self.slot_size = slot_size
        self.rel_size = rel_size
        self.rnn_hid = slot_size
        self.num_slot = num_slot
        self.step = step
        self.rd = rd
        self.learn_init_mem = learn_init_mem
        self.output_size =output_size
        self.out_att_size = out_att_size

        self.qkv_projector = nn.ModuleList([nn.Linear(slot_size, num_slot*3)]*step)
        self.qkv_layernorm = nn.ModuleList([nn.LayerNorm([num_slot*3])]*step)

        if init_alphas[0] is None:
            self.alpha1 = [nn.Parameter(torch.zeros(1))] * step
            for ia, a in enumerate(self.alpha1):
                setattr(self, 'alpha1' + str(ia), self.alpha1[ia])
        else:
            self.alpha1 = [init_alphas[0]]* step

        if init_alphas[1] is None:
            self.alpha2 = [nn.Parameter(torch.zeros(1))] * step
            for ia, a in enumerate(self.alpha2):
                setattr(self, 'alpha2' + str(ia), self.alpha2[ia])
        else:
            self.alpha2 = [init_alphas[1]] * step

        if init_alphas[2] is None:
            self.alpha3 = [nn.Parameter(torch.zeros(1))] * step
            for ia, a in enumerate(self.alpha3):
                setattr(self, 'alpha3' + str(ia), self.alpha3[ia])
        else:
            self.alpha3 = [init_alphas[2]] * step


        self.input_projector = MLP(input_size, slot_size, hid_dim=mlp_hid)
        self.input_projector2 = MLP(input_size, slot_size, hid_dim=mlp_hid)
        self.input_projector3 = MLP(input_size, num_slot, hid_dim=mlp_hid)


        self.input_gate_projector = nn.Linear(self.slot_size, self.slot_size*2)
        self.memory_gate_projector = nn.Linear(self.slot_size, self.slot_size*2)
        # trainable scalar gate bias tensors
        self.forget_bias = nn.Parameter(torch.tensor(1., dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(0., dtype=torch.float32))

        self.rel_projector = nn.Linear(slot_size*slot_size, rel_size)
        self.rel_projector2 = nn.Linear(num_slot * slot_size, slot_size)
        self.rel_projector3 = nn.Linear(num_slot * rel_size, out_att_size)
        self.layernorm2 = nn.LayerNorm([out_att_size])

        self.mlp = nn.Sequential(
            nn.Linear(out_att_size, self.mlp_size),
            nn.ReLU(),
           nn.Linear(self.mlp_size, self.mlp_size),
           nn.ReLU(),
        )

        self.out = nn.Linear(self.mlp_size, output_size)

        if self.learn_init_mem:
            if torch.cuda.is_available():
                self.register_parameter('item_memory_state_bias',
                                        torch.nn.Parameter(torch.Tensor(self.slot_size, self.slot_size).cuda()))
                self.register_parameter('rel_memory_state_bias', torch.nn.Parameter(
                    torch.Tensor(self.num_slot, self.slot_size, self.slot_size).cuda()))

            else:
                self.register_parameter('item_memory_state_bias',
                                        torch.nn.Parameter(torch.Tensor(self.slot_size, self.slot_size)))
                self.register_parameter('rel_memory_state_bias',
                                        torch.nn.Parameter(torch.Tensor(self.num_slot, self.slot_size, self.slot_size)))

            stdev = 1 / (np.sqrt(self.slot_size + self.slot_size))
            nn.init.uniform_(self.item_memory_state_bias, -stdev, stdev)
            stdev = 1 / (np.sqrt(self.slot_size + self.slot_size + self.num_slot))
            nn.init.uniform_(self.rel_memory_state_bias, -stdev, stdev)


    def create_new_state(self, batch_size, gpu_id=0):
        if self.learn_init_mem:
            read_heads = torch.zeros(batch_size, self.output_size)
            item_memory_state = self.item_memory_state_bias.clone().repeat(batch_size, 1, 1)
            rel_memory_state = self.rel_memory_state_bias.clone().repeat(batch_size, 1, 1, 1)
            if torch.cuda.is_available() and gpu_id>=0:
                read_heads = read_heads.cuda()
        else:

            item_memory_state =  torch.stack([torch.zeros(self.slot_size, self.slot_size) for _ in range(batch_size)])
            read_heads =  torch.zeros(batch_size, self.output_size)
            rel_memory_state =  torch.stack([torch.zeros(self.num_slot, self.slot_size, self.slot_size) for _ in range(batch_size)])
            if torch.cuda.is_available() and gpu_id>=0:
                item_memory_state = item_memory_state.cuda()
                read_heads = read_heads.cuda()
                rel_memory_state = rel_memory_state.cuda()

        return read_heads, item_memory_state, rel_memory_state



    def compute_gates(self, inputs, memory):

        memory = torch.tanh(memory)
        if len(inputs.shape) == 3:
            if inputs.shape[1] > 1:
                raise ValueError(
                    "input seq length is larger than 1. create_gate function is meant to be called for each step, with input seq length of 1")
            inputs = inputs.view(inputs.shape[0], -1)

            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(dim=1)
            gate_memory = self.memory_gate_projector(memory)
        else:
            raise ValueError("input shape of create_gate function is 2, expects 3")

        gates = gate_memory + gate_inputs
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]

        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def compute(self, input_step, prev_state):

        hid = prev_state[0]
        item_memory_state = prev_state[1]
        rel_memory_state = prev_state[2]

        # transform input
        controller_outp = self.input_projector(input_step)
        if rel_memory_state:
            controller_outp2 = self.input_projector2(input_step)
            controller_outp3 = self.input_projector3(input_step)


            # Mr read
            controller_outp3 = F.softmax(controller_outp3, dim=-1)
            controller_outp4 = torch.einsum('bn,bd,bndf->bf', controller_outp3, controller_outp2, rel_memory_state)
            X2 = torch.einsum('bd,bf->bdf', controller_outp4, controller_outp2)

        # Mi write
        X = torch.matmul(controller_outp.unsqueeze(2), controller_outp.unsqueeze(1))  # Bxdxd
        input_gate, forget_gate = self.compute_gates(controller_outp.unsqueeze(1), item_memory_state)

        if self.rd:
            # Mi write gating
            R = input_gate * F.tanh(X)
            R += forget_gate * item_memory_state
        else:
            # Mi write
            R = item_memory_state + torch.matmul(controller_outp.unsqueeze(2), controller_outp.unsqueeze(1))  # Bxdxd

        for i in range(self.step):
            #SAM
            if rel_memory_state:
                qkv = self.qkv_projector[i](R+self.alpha2[i]*X2)
            else:
                qkv = self.qkv_projector[i](R)

            qkv = self.qkv_layernorm[i](qkv)
            qkv = qkv.permute(0,2,1) #Bx3Nxd

            q,k,v = torch.split(qkv, [self.num_slot]*3, 1)#BxNxd


            R0 = op_att(q, k, v) #BxNxdxd

            #Mr transfer to Mi
            R2= self.rel_projector2(R0.view(R0.shape[0], -1, R0.shape[3]).permute(0, 2, 1))
            R =  R + self.alpha3[i] * R2

            #Mr write
            if rel_memory_state:
                rel_memory_state = self.alpha1[i]*rel_memory_state + R0
            else:
                rel_memory_state = R0

        #Mr transfer to output
        r_vec = self.rel_projector(rel_memory_state.view(rel_memory_state.shape[0],
                                                         rel_memory_state.shape[1],
                                                         -1)).view(input_step.shape[0],-1)
        out = self.rel_projector3(r_vec)
        out = self.layernorm2(out)


        return out, (out, R, rel_memory_state)

    def forward(self, input_step, hidden=None):

        if len(input_step.shape)==3:
            self.init_sequence(input_step.shape[1])
            for i in range(input_step.shape[0]):
                logit, self.previous_state = self.compute(input_step[i], self.previous_state)

        else:
            if hidden is not None:
                logit, self.previous_state  = self.compute(input_step, hidden)
            else:
                logit, self.previous_state = self.compute(input_step,  self.previous_state)
        mlp = self.mlp(logit)
        out = self.out(mlp)
        self.previous_state = (out, self.previous_state[1], self.previous_state[2])

        return out, self.previous_state

    def init_sequence(self, batch_size, gpu_id=0):
        """Initializing the state."""
        self.previous_state = self.create_new_state(batch_size, gpu_id)
        return self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

if __name__ == "__main__":

    N=64
    S=80
    B=32
    K = torch.ones((B, S, N))
    V = torch.ones((B, S, N))
    q = torch.ones((B, N))
    R = op_att(K,V,q)
    print(R.shape)

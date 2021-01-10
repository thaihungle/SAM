from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import norm_col_init, weights_init
from .stm_rl import  STM

class A3CSAM(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3CSAM, self).__init__()
        self.input_shape = num_inputs
        self.num_channel = num_inputs[0]
        self.noise = nn.Dropout(0.9)


        self.conv1 = nn.Conv2d(self.num_channel, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        self.sam = STM(256, 256,
                            num_slot=8, mlp_size=128,
                            slot_size=96, out_att_size=64, step=1,
                            rel_size=96, rd=True)
        self.lstm = nn.LSTMCell(1024, 256)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)



        self.train()

    def forward(self, inputs):
        inputs, (hx,cx, h2, c2) = inputs
        if self.input_shape[0] == 3 and self.input_shape[1] == 12:
            inputs = inputs.permute(0, 3, 1, 2)
            x = F.relu(self.maxp3(self.conv3(inputs)))
            x = F.relu(self.maxp4(self.conv4(x)))
        elif self.input_shape[1] == 19:
            # inputs = inputs.permute(0, 3, 1, 2)
            x = F.relu(self.conv3(inputs))
            x = F.relu(self.conv4(x))
        else:
            x = F.relu(self.maxp1(self.conv1(inputs)))
            x = F.relu(self.maxp2(self.conv2(x)))
            x = F.relu(self.maxp3(self.conv3(x)))
            x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        #x = self.noise(x)

        # print(hx.shape)
        # print(cx.shape)
        h2, c2 = self.lstm(x, (h2,c2))
        out, (hx,cx,_) = self.sam(h2, (hx,cx,None))
        # h2,c2 = self.lstm(torch.cat([x, out], dim=-1), (h2,c2))

        x = torch.cat([out,h2], dim=-1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx, h2, c2)

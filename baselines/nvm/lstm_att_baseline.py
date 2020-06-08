import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, emb=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        if emb:
            self.embedding = nn.Embedding(input_size, hidden_size)
        else:
            self.embedding = nn.Linear(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output.unsqueeze(0), hidden)
        return output.squeeze(0), hidden

    def initHidden(self, bs):
        return torch.zeros(1, bs, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1, emb=False, max_length=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        if emb:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = nn.Linear(output_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=False)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
        # print(attn_weights.shape)
        # print(encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights[:,:encoder_outputs.shape[0]].unsqueeze(1),
                                 encoder_outputs.permute(1,0,2)).squeeze(1)

        output = torch.cat((embedded, attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self, bs):
        return torch.zeros(1, bs, self.hidden_size, device=device)


class AttnEncoderDecoder(nn.Module):

    def __init__(self, indim, odim, hidden_size, max_att_len=100, dropout_p=0.1, emb_enc=False, emb_dec=False):
        super(AttnEncoderDecoder, self).__init__()

        self.encoder = EncoderRNN(indim, hidden_size,emb_enc)
        self.decoder = AttnDecoderRNN(odim, hidden_size, dropout_p, emb_dec, max_length=max_att_len)


    def forward(self, encode_inputs, target_length, constant_decode_input=True):
        encoder_outputs = torch.zeros(encode_inputs.shape[0],encode_inputs.shape[1], self.encoder.hidden_size, device=device)

        for ei in range(encode_inputs.shape[0]):
            encoder_output, self.previous_state = self.encoder(
                encode_inputs[ei], self.previous_state)
            encoder_outputs[ei] = encoder_output

        decoder_input = torch.zeros(encode_inputs.shape[1], self.decoder.output_size, device=device)

        decoder_hidden = self.previous_state

        decoder_outputs = torch.zeros(target_length, encode_inputs.shape[1], self.decoder.output_size, device=device)

        if constant_decode_input:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_outputs[di], decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def create_new_state(self, batch_size):
        return self.encoder.initHidden(batch_size)

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.previous_state = self.create_new_state(batch_size)

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

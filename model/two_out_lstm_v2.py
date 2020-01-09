"""
Implementation of LSTM module for synchronous Forwards Backward RNN
"""
import torch
import torch.nn as nn



class TwoOutLSTM_v2(nn.Module):

    def __init__(self, input_dim=110, hidden_dim=256, layers=2):
        super(TwoOutLSTM_v2, self).__init__()

        # Dimensions
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = input_dim // 2

        # Number of LSTM layers
        self._layers = layers

        # LSTM
        self._lstm = nn.LSTM(input_size=self._input_dim, hidden_size=self._hidden_dim, num_layers=layers, dropout=0.3)

        # All weights initialized with xavier uniform
        nn.init.xavier_uniform_(self._lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self._lstm.weight_ih_l1)
        nn.init.orthogonal_(self._lstm.weight_hh_l0)
        nn.init.orthogonal_(self._lstm.weight_hh_l1)

        # Bias initialized with zeros expect the bias of the forget gate
        self._lstm.bias_ih_l0.data.fill_(0.0)
        self._lstm.bias_ih_l0.data[self._hidden_dim:2*self._hidden_dim].fill_(1.0)

        self._lstm.bias_ih_l1.data.fill_(0.0)
        self._lstm.bias_ih_l1.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._lstm.bias_hh_l0.data.fill_(0.0)
        self._lstm.bias_hh_l0.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._lstm.bias_hh_l1.data.fill_(0.0)
        self._lstm.bias_hh_l1.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)


        # Batch normalization (Weights initialized with one and bias with zero)
        self._norm_0 = nn.LayerNorm(self._input_dim, eps=.001)
        self._norm_1 =nn.LayerNorm(self._hidden_dim, eps=.001)

        # Linear layer
        self._wlinear = nn.Linear(self._hidden_dim, self._input_dim)
        nn.init.xavier_uniform_(self._wlinear.weight)
        self._wlinear.bias.data.fill_(0.0)


    def _init_hidden(self, batch_size, device):
        '''Initialize hidden states
        :return: new hidden state
        '''

        return (torch.zeros(self._layers, batch_size, self._hidden_dim).to(device),
                torch.zeros(self._layers, batch_size, self._hidden_dim).to(device))

    def new_sequence(self, batch_size=1, device="cpu"):
        '''Prepare model for a new sequence'''
        self._hidden = self._init_hidden(batch_size, device)
        return

    def check_gradients(self):
        print('Gradients Check')
        for p in self.parameters():
            print(p.grad.shape)
            print(p.grad.data.norm(2))

    def forward(self, input):
        '''Forward computation
        :param input:           tensor( sequence length, batch size, encoding size)
        :return forward:      forward prediction (batch site, encoding size)
                back:         backward prediction (batch size, encoding size)
        '''

        # Normalization over encoding dimension
        norm_0 = self._norm_0(input)

        # Compute LSTM unit
        out, self._hidden = self._lstm(norm_0, self._hidden)

        # Normalization over hidden dimension
        norm_1 = self._norm_1(out)

        # Linear layer
        lin_out = self._wlinear(norm_1)

        # Split into forward and backward prediction
        [forward, back] = torch.split(lin_out, self._output_dim, dim=-1)


        return forward, back

import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(input_size[-1], hidden_size[-1])
        self.linear2 = nn.Linear(hidden_size[-1], output_size[-1])

    def forward(self, obs):
        # evaluate q values
        # i = obs['obs'][:,:,:].float()
        i = obs.float()
        x = F.relu(self.linear1(i))
        # print("DQN dimension i: ",i.shape," and of x: ", x.shape)
        # q = F.relu(self.linear2(x))
        q = self.linear2(x)
        return q


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RNN, self).__init__()

        self.linear1 = nn.Linear(input_size[-1], hidden_size[-1])
        # gru expects [batch_size, seq_len, features] as input
        # is it possible to reshape from (batch size, seq_len, n_networks, n_nodes, node_features)
        # to (batch size, seq_len, all_features)? Alternatively change the memory structure?
        #self.rearrange = Rearrange('')
        self.gru = nn.GRU(input_size=hidden_size[-1], hidden_size=hidden_size[-1], batch_first=True)
        self.linear2 = nn.Linear(hidden_size[-1], output_size[-1])

    def reset_hidden(self):
        self.hidden = None
    def forward(self, obs):
        # evaluate q values
        i = obs.float()
        print("RNN dimension of i: ", i.shape)
        x = F.relu(self.linear1(i))
        print("RNN dimension of x: ", x.shape)
        h, self.hidden = self.gru(x, self.hidden)
        q = self.linear2(h)
        return q


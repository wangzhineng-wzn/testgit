import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import torch
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))
        u = u / self.scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        output = torch.bmm(attn, v)

        return attn, output

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)
        output = self.fc_o(output)

        return attn, output

class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return attn, output

# CNN
class CNN(nn.Module):
    def __init__(self, L):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32*int(L/4)*int(L/4), 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

# CNN_3layers
class CNN_3layers(nn.Module):
    def __init__(self, L):
        super(CNN_3layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(576, 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv3(x))
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

# CLSGCN
class CLSGCN(nn.Module):
    def __init__(self, L):
        super(CLSGCN, self).__init__()
        self.lstm = nn.LSTM(4, 32, num_layers=1, bias=True, batch_first=True)
        self.cnn = CNN(L)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        self.gcn = GraphConvolution(in_features=64, out_features=128)
        self.attention = SelfAttention(n_head=1, d_k=32, d_v=16, d_x=32, d_o=32)
        self.flatten = nn.Flatten()

    def forward(self, x1, x2, adj):
        x1 = self.cnn(x1)
        x2 = self.lstm(x2)[0]
        x2 = x2[:,-1,:]
        x1 = x1[:,np.newaxis,:]
        x2 = x2[:,np.newaxis,:]
        x = torch.cat([x1, x2], dim=1)
        x = self.attention(x)[1]
        x = self.flatten(x)
        x = self.gcn(x, adj)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

# CLSGCN_3CNN
class CLSGCN_3CNN(nn.Module):
    def __init__(self, L):
        super(CLSGCN_3CNN, self).__init__()
        self.lstm = nn.LSTM(4, 32, num_layers=1, bias=True, batch_first=True)
        self.cnn = CNN_3layers(L)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        self.gcn = GraphConvolution(in_features=64, out_features=128)
        self.attention = SelfAttention(n_head=1, d_k=32, d_v=16, d_x=32, d_o=32)
        self.flatten = nn.Flatten()

    def forward(self, x1, x2, adj):
        x1 = self.cnn(x1)
        x2 = self.lstm(x2)[0]
        x2 = x2[:,-1,:]
        x1 = x1[:,np.newaxis,:]
        x2 = x2[:,np.newaxis,:]
        x = torch.cat([x1, x2], dim=1)
        x = self.attention(x)[1]
        x = self.flatten(x)
        x = self.gcn(x, adj)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x
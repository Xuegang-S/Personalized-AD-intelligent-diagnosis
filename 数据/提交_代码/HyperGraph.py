import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Parameter

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

def generate_G_from_H(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :return: G
    """
    H = np.array(H)                     # 将输入的超图关联矩阵 H 转换为 NumPy 数组，以便进行后续的矩阵操作。
    n_edge = H.shape[1]                 # 超图的边数
    # the weight of the hyperedge
    W = np.ones(n_edge)                 # 超图的权重矩阵，初始化为1
    # the degree of the node
    DV = np.sum(H * W, axis=1)          # 结点的度矩阵，每个结点的度为与其关联的超边的权重之和
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)              # 超边的度矩阵，每个超边的度为与其关联的结点的个数

    invDE = np.mat(np.diag(np.power(DE, -1)))     # 超边的度矩阵的逆矩阵，其中每个对角元素是对应超边的度的倒数
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))     # 结点的度矩阵的平方根的逆矩阵，其中每个对角元素是对应结点的度的平方根的倒数
    W = np.mat(np.diag(W))                        # 超图的权重矩阵
    H = np.mat(H)                                 # 超图的关联矩阵
    HT = H.T                                      # 超图的关联矩阵的转置

    G = DV2 * H * W * invDE * HT * DV2
    # G = DV2 @ H @ W @ invDE @ HT @ DV2
    G = torch.FloatTensor(G)
    return G

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x

x = torch.randn(10, 10)
adj = torch.randn(10, 10)       # 注意，adj每行不能全为0
G = generate_G_from_H(adj)
model = HGNN(in_ch=x.shape[1], n_class=2, n_hid=10)
output = model(x, G)
print(output.shape)
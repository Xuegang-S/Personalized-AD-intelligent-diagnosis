import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.registry import register_model


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha,
                 concat=True):  # in_features=1433，特征的个数。out_features = 8
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # self.W是一个形状为[1433, 8]的矩阵
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化的一种方法
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # a的形状为[16, 1]
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 对a初始化

        self.leakyrelu = nn.LeakyReLU(self.alpha)  # 就是一个激活函数

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)  # torch.ones_like(e): 创造和e形状相同的全1矩阵。 将没有连接的便设置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)
        # torch.where(condition, x, y) -> Tensor, condition是一个bool型的矩阵，x和y是两个矩阵，当condition中的元素为True时，输出x中的元素，否则输出y中的元素.
        # adj是一个邻接矩阵，当两个节点之间有连接时，adj>0，否则adj=0. 两个节点之间有连接时，attention中的元素为e中的元素，否则为负无穷。
        attention = F.softmax(attention, dim=1)  # softmax()操作，注意力系数
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # 根据注意力系数，对特征就行加权求和

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # a也是一个可学习的参数
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T  # e是一个矩阵，记录了eij
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] # nfeat是特征个数
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


x = torch.randn(32, 1433)
adj = torch.randn(32, 32)
model = GAT(nfeat=x.shape[1],
            nhid=8,
            nclass=2,
            dropout=0.5,
            alpha=0.2,
            nheads=8)
output = model(x, adj)
print(output.shape)
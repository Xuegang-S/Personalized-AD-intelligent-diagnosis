import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.rand(5, 9)
adj = torch.rand(5, 5)

def norm_laplacian(adj):
    adj = adj + torch.eye(adj.size(0))
    degree = torch.sum(adj, dim=1)
    D = torch.diag(torch.pow(degree, -0.5))
    return D.mm(adj).mm(D)

lap_adj = norm_laplacian(adj)
eng_val, eng_vec = torch.linalg.eig(lap_adj)

sign_flip = torch.rand(eng_vec.size(1))
sign_flip[sign_flip >= 0.5] = 1.0
sign_flip[sign_flip < 0.5] = -1.0
eng_vec = eng_vec * sign_flip.unsqueeze(0)

class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, use_bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3)


    def forward(self, h):
        qkv = self.qkv(h)
        qkv = qkv.reshape(h.shape[0], 3, self.num_heads, self.embed_dim // self.num_heads)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        qk = torch.matmul(q, k.permute(0, 2, 1)) / (self.embed_dim ** 0.5)
        qk = torch.nn.functional.softmax(qk, dim=-1)
        qkv = torch.matmul(qk, v)
        qkv = qkv.permute(1, 0, 2)
        qkv = qkv.reshape(h.shape[0], self.embed_dim)

        return qkv


class GraphTransformerLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_bias = use_bias

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.attention = MultiHeadAttentionLayer(in_dim, num_heads, use_bias)

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

    def forward(self, h):
        h_in1 = h
        h_attention_out = self.attention(h)

        h = F.dropout(h_attention_out, self.dropout, training=self.training)
        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h

        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)


        return h


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, pos_enc_dim, num_class, num_heads, dropout=0.0,
                 n_layers=1, layer_norm=None, batch_norm=True, residual=True, use_bias=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_bias = use_bias


        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, self.dropout,
                                                           self.layer_norm, self.batch_norm, self.residual,
                                                           self.use_bias) for _ in range(n_layers-1)])

        self.classifier = nn.Linear(hidden_dim, num_class)

    def forward(self, x, lap_pos_enc):
        h = self.embedding_h(x)
        h_lap_pos_enc = self.embedding_lap_pos_enc(lap_pos_enc.float())
        h = h + h_lap_pos_enc

        for layer in self.layers:
            h = layer(h)

        h = self.classifier(h)

        return h



model = GraphTransformer(in_dim=x.shape[1],
                         hidden_dim=6,
                         pos_enc_dim=x.shape[0],
                         num_class=2,
                         num_heads=3,
                         dropout=0.0,
                         n_layers=4,
                         layer_norm=False,
                         batch_norm=True,
                         residual=True,
                         use_bias=True)
print(model)
y = model(x, eng_vec)
print(y.shape)

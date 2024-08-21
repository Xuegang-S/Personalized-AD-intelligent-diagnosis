from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import math
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from pygcn.classification_metrics import all_metrics
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio
# from pygcn.layers import GraphConvolution
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--classes', type=list, default=[2, 3],)
parser.add_argument('--adj_way', type=str, default='eye')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)    # Torch.spmm只支持 sparse 在前，dense 在后的矩阵乘法
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Graph_Transformer_multi_channel(nn.Module):
    def __init__(self, nfeat1, nfeat2, nfeat3, nhid, nclass, dropout):
        super(Graph_Transformer_multi_channel, self).__init__()

        self.dropout = dropout
        self.feanum = 24
        self.num_heads = 1
        self.embed_dim = self.feanum
        self.qkv = torch.nn.Linear(self.feanum, self.feanum * 3)
        self.gc1 = GraphConvolution(nfeat1, 8)
        self.gc2 = GraphConvolution(nfeat2, 8)
        self.gc3 = GraphConvolution(nfeat3, 8)
        self.liner1 = torch.nn.Linear(84, 16)
        self.liner2 = torch.nn.Linear(16, nclass)


    def forward(self, x1,x2,x3, adj):

        x11 = F.relu(self.gc1(x1, adj))
        x21 = F.relu(self.gc2(x2, adj))
        x31 = F.relu(self.gc3(x3, adj))
        x = torch.cat((x11, x21, x31), 1)

        qkv = self.qkv(x)  # qkv.shape = (num_nodes+1, embed_dim*3)
        qkv = qkv.reshape(x.shape[0], 3, self.num_heads,
                          self.embed_dim // self.num_heads)  # qkv.shape = (num_nodes+1, 3, num_heads, embed_dim//num_heads)
        qkv = qkv.permute(1, 0, 2, 3)  # qkv.shape = (3, num_nodes+1, num_heads, embed_dim//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q.shape = (num_nodes+1, num_heads, embed_dim//num_heads)
        qk = torch.matmul(q.permute(1, 2, 0), k.permute(1, 0, 2)) / (
                    self.embed_dim ** 0.5)  # qk.shape = (num_nodes+1, num_heads, num_nodes+1)
        qk = torch.nn.functional.softmax(qk, dim=-1)  # qk.shape = (num_nodes+1, num_heads, num_nodes+1)
        qkv = torch.matmul(qk, v.permute(1, 2, 0))  # qkv.shape = (num_nodes+1, num_heads, embed_dim//num_heads)
        qkv = qkv.permute(2, 0, 1)  # qkv.shape = (num_heads, num_nodes+1, embed_dim//num_heads)
        qkv = qkv.reshape(x.shape[0], self.embed_dim)  # qkv.shape = (num_nodes+1, embed_dim)
        # x = qkv
        x = torch.cat((x1,x2,x3,qkv), 1)
        x = F.relu(self.liner1(x))
        x = F.relu(self.liner2(x))

        return x
def feature_selection(matrix, labels, fnum):
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step = 1)
    featureX = matrix
    featureY = labels
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    feature_index = selector.get_support(True)
    return x_data, feature_index

class CrossEntropyLoss_personalized(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_personalized, self).__init__()
    def forward(self, predict, target, adj, idx_train, idx_val):
        adj = adj.reshape(-1,1)
        eps = 1e-12
        loss = -1. * predict.gather(1, target.unsqueeze(-1)).reshape(-1, 1) + torch.log(
            torch.exp(predict + eps).sum(dim=1)).reshape(-1, 1)
        loss = loss * adj

        return loss.mean()

def train(epoch,idx_train,idx_val):
    model.train()
    optimizer.zero_grad()
    output = model(features_SMRI,features_Atrophy,feature_SNP, adj)
    loss_train = criterion(output[idx_train], label[idx_train], adj_pheno, idx_train, idx_val)
    acc_train = output[idx_train].max(1)[1].eq(label[idx_train]).sum().item() / len(idx_train)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features_SMRI,features_Atrophy,feature_SNP, adj)

    loss_val = criterion(output[idx_val], label[idx_val], adj_pheno, idx_train, idx_val)
    acc_val = output[idx_val].max(1)[1].eq(label[idx_val]).sum().item() / len(idx_val)

    return loss_train, acc_train, loss_val, acc_val
def test(epoch,idx_test):
    model.eval()
    output = model(features_SMRI,features_Atrophy,feature_SNP, adj)
    y_score = F.softmax(output[idx_test], dim=1)
    return label[idx_test].cpu().detach().numpy(), y_score.cpu().detach().numpy()
# Train model
t_total = time.time()
label_total, y_total = [], []
wrong_id = []
label_train_check, y_test_train_check = [], []
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut() # 实例化LOO对象
a=0
num_right, num_wrong = 0, 0
t_start = time.time()

data = sio.loadmat('C:\\Users\\John\\Desktop\\code\\pygcn1\\output_data\\NCAD.mat')  ## NCMCI, NCAD, MCIAD
features_SMRI = data['features_SMRI']
features_Atrophy = data['features_Atrophy']
feature_SNP = data['feature_SNP']
adj_race = data['adj_race']
adj_apoe = data['adj_apoe']
adj_edu = data['adj_edu']
adj_sex = data['adj_sex']
adj_age = data['adj_age']
adj = data['adj']
[label] = torch.LongTensor(data['label'].astype(int))
features_SMRI = StandardScaler().fit_transform(features_SMRI)
features_Atrophy = StandardScaler().fit_transform(features_Atrophy)
feature_SNP = StandardScaler().fit_transform(feature_SNP)
features = np.hstack([features_SMRI, features_Atrophy, feature_SNP])

for train_index, test_index in loo.split(features):

    features_SMRI, feature_index = feature_selection(features_SMRI, label, 40)
    features_Atrophy, feature_index = feature_selection(features_Atrophy, label, 10)
    feature_SNP, feature_index = feature_selection(feature_SNP, label, 10)
    features = np.hstack([features_SMRI, features_Atrophy, feature_SNP])

    features_SMRI = torch.FloatTensor(features_SMRI)
    features_Atrophy = torch.FloatTensor(features_Atrophy)
    feature_SNP = torch.FloatTensor(feature_SNP)
    features = torch.FloatTensor(features)

    # Calculate all pairwise distances
    distv = distance.pdist(features, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    adj_corr = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj_pheno = adj_race + adj_apoe + adj_edu + adj_sex + adj_age   ## 2-3 任务 + 1

    adj = torch.tensor(adj_corr, dtype=torch.float32)
    adj_corr = torch.tensor(adj_corr, dtype=torch.float32)
    adj_pheno = torch.tensor(adj_pheno, dtype=torch.float32)

    model = Graph_Transformer_multi_channel(nfeat1=features_SMRI.shape[1],nfeat2=features_Atrophy.shape[1],nfeat3=feature_SNP.shape[1],
                nhid=args.hidden,
                nclass=label.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss_personalized()

    aa=0
    adj = torch.FloatTensor(adj)
    adj_pheno = torch.Tensor([adj_pheno[int(test_index)][idx] for idx in train_index])
    acc_train = []
    for epoch in range(args.epochs):
        loss_train, acc_train, loss_val, acc_val = train(epoch,train_index,test_index)
        max_acc = acc_val
    label_test, y_test = test(epoch, test_index)
    for i in train_index:
        ii = [i]
        label_train_check, y_test_train_check = test(epoch, ii)
        y_test_train_check = np.argmax(y_test_train_check, axis=1)
        if label_train_check == y_test_train_check:
            a =1
        else:
            if wrong_id == []:
                wrong_id = i
            else:
                wrong_id = np.hstack([wrong_id, i])
    if label_total == []:
        label_total = label_test
        y_total = y_test
    else:
        if np.isnan(y_test[0,0]):
            print('test_index.isnan', test_index, 'number_isnan', aa)
            aa = aa+1
        else:
            label_total = np.hstack([label_total, label_test])
            y_total = np.vstack([y_total, y_test])
    y_test0 = np.argmax(y_test, axis=1)
    if label_test == y_test0:
        num_right = num_right + 1
        print('right', test_index,label_test,y_test0,'acc_train', acc_train,'acc_right_now', 'acc: {:.4f}'.format((num_right/(num_right+num_wrong))))
    else:
        num_wrong = num_wrong + 1
        print('wrong', test_index,label_test,y_test0,'acc_train', acc_train,'acc_right_now','acc: {:.4f}'.format((num_right/(num_right+num_wrong))),"y_pred",y_test,"label_test",label_test)
t_end = time.time()
t_consume = (t_end-t_start)/60
print('number_isnan', aa)
y_total0 = np.argmax(y_total, axis=1)
right, wrong, right_0, right_1 = 0, 0, 0, 0
for i in range(np.asarray(y_total0).shape[0]):
    if label_total[i]==y_total0[i]:
        right=right+1
        if label_total[i]==0:
            right_0 = right_0 + 1
        else:
            right_1 = right_1 + 1
    else:
        wrong = wrong+1
ACC1 = right/label_total.shape[0]
SEN1 = right_0/(label_total.shape[0]-np.count_nonzero(label_total))
SPE1 = right_1/np.count_nonzero(label_total)
AUC1 = roc_auc_score(label_total, y_total[:,1])
print('acc_sxg:', 'acc: {:.4f}'.format(ACC1),'sen: {:.4f}'.format(SEN1),'spec: {:.4f}'.format(SPE1),'auc: {:.4f}'.format(AUC1))
acc, pre, sen, f1, spec, kappa, auc, qwk = (all_metrics(label_total, y_total))
print('results:', 'acc: {:.4f}'.format(acc),'sen: {:.4f}'.format(sen),'spec: {:.4f}'.format(spec),'auc: {:.4f}'.format(auc))
print('time--consume', t_consume)
from collections import Counter
result = Counter(wrong_id)
d = sorted(result.items(), key=lambda x: x[1], reverse=True)
print(d)
print('wrong_id.shape', wrong_id.shape[0])

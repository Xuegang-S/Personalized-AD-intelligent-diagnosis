from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.classification_metrics import all_metrics
from pygcn.utils import load_data,load_data11,load_data1,accuracy,construct_adj_cosine,construct_adj_knn_cosine,construct_adj_Gaussian,construct_adj_Euclid,construct_adj_knn,construct_adj_Gaussian,construct_adj_Gaussian_theta, construct_adj_Gaussian_knn
from pygcn.models import GCN,MLP
from scipy.spatial import distance
print('label_total.shape',1111111)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--classes', type=list, default=[0, 2],)
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

# Load data
adj1, adj2, features, label, index_train, index_val, index_test = load_data(args) ## adj单位矩阵，adj2考虑年龄
print('adj.shape',adj1.shape)
print('adj2.shape',adj2.shape)

# Calculate all pairwise distances
distv = distance.pdist(features, metric='correlation')
dist = distance.squareform(distv)
sigma = np.mean(dist)
adj_corr = np.exp(- dist ** 2 / (2 * sigma ** 2))
# adj = adj + np.eye(adj.shape[0])
# adj = torch.FloatTensor(adj)
adj1 = np.eye(adj1.shape[0])
adj1 = torch.FloatTensor(adj1)
# Model and optimizer
# model = GCN(nfeat=features.shape[1],
#             nhid=args.hidden,
#             nclass=label.max().item() + 1,
#             dropout=args.dropout)
# optimizer = optim.Adam(model.parameters(),
#                        lr=args.lr, weight_decay=args.weight_decay)
# criterion = torch.nn.CrossEntropyLoss()

labels = label
idx_train = index_train
idx_val = index_val
idx_test = index_test
max_acc = 0.0
print('label_total.shape',2222222)
t0 = time.time()

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
def feature_selection(matrix, labels, fnum):
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step = 1)
    featureX = matrix
    featureY = labels
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    feature_index = selector.get_support(True)
    feature_important = selector.ranking_ ## # 打印的是相应位置上属性的重要性排名
    feature_rank = np.argsort(feature_important)  # 按升序排列

    return x_data, feature_index,feature_important,feature_rank
def train(epoch,idx_train,idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = output[idx_train].max(1)[1].eq(labels[idx_train]).sum().item() / len(idx_train)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = criterion(output[idx_val], labels[idx_val])
    acc_val = output[idx_val].max(1)[1].eq(labels[idx_val]).sum().item() / len(idx_val)

    return loss_train, acc_train, loss_val, acc_val
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))


def test(epoch,idx_test):
    model.eval()
    output = model(features, adj)
    y_score = F.softmax(output[idx_test], dim=1)
    return labels[idx_test].cpu().detach().numpy(),y_score.cpu().detach().numpy()

# Train model
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
t_total = time.time()
label_total, y_total = [], []
from sklearn.model_selection import LeaveOneOut
print('features.shape', features.shape)
print('label.shape', label.shape)

loo = LeaveOneOut() # 实例化LOO对象
# for train_index, test_index in loo.split(features):
#     print('test_index', test_index)
t_start = time.time()
# for m6 in range(2,3):  # 2-9
#     for m5 in range(2,3):
#         for m4 in range(2,3):
#             for m3 in range(2,9):
#                 for m2 in range(2,9):
#                     for m1 in range(2, 9):  # 3 - 7
features0, features_age_sex, feature_statics, feature_statics2, feature_SNP, feature_PRS, feature_RaceEduApoe, adj, adj_sex, adj_age, adj_race, adj_apoe, adj_edu, label, label_age, index_train, index_val, index_test = load_data1(
    args)
features = np.hstack([features0, feature_statics2, feature_SNP])
# features = feature_statics2
print('features0.shap', features0.shape,'feature_statics2.shap', feature_statics2.shape,'feature_SNP.shap', feature_SNP.shape,'features.shap', features.shape)

########################################
v1 = features[label==0,:]
v2 = features[label==1,:]
# print('v1.shap', v1.shape,'v2.shap', v2.shape,'features.shap', features.shape)
import numpy as np
from scipy.stats import ttest_ind

res = ttest_ind(v1, v2).pvalue
res_rank = np.argsort(res)  # 按升序排列
# print('P values-shape',res.shape)
# print('P values',res)
print('res_rank_ttest',res_rank[0:20])

#########################################

# print('features.shape', features.shape)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(features, label)
importances = rf.feature_importances_
# print('importances',importances)
import matplotlib.pyplot as plt
res_rank2 = np.argsort(-importances)  # 按降序排列
print('res_rank_RF',res_rank2[0:20])
# plt.bar(range(features.shape[1]), importances)
# plt.xlabel('Feature Index')
# plt.ylabel('Feature Importance')
# plt.show()
#########################################

features, feature_index,feature_important,feature_rank = feature_selection(features, label, 60)
# features = features0 #
# features = feature_statics2
# features = np.hstack([features0, feature_statics2])
# features, feature_index,feature_important,feature_rank = feature_selection(features, label, 20)

# features = features[:,res_rank[0:50]]
# print('features0.shap', features0.shape,'feature_statics2.shap', feature_statics2.shape,'feature_SNP.shap', feature_SNP.shape)
# print(feature_index.shape)
# print('feature_index', feature_index)
# print('feature_important', feature_important)
# print('feature_rank', feature_rank)
a1,a2,a3 = 0,0,0
m=60
for i in range(m):
    if res_rank[i]<251:  ## res_rank   feature_index
        a1 = a1 + 1
    if res_rank[i] > 251 and res_rank[i]< 275:
        a2 = a2+ 1
    if res_rank[i] > 275:
        a3 = a3 + 1
print('feature_num',a1,'statastic_num',a2,'SNP_num',a3)
print( 'feature_num: {:.1f}'.format(100*a1/m),'statastic_num: {:.1f}'.format(100*a2/m),'SNP_num: {:.1f}'.format(100*a3/m))

for train_index, test_index in skf.split(features, label):
    features = StandardScaler().fit_transform(features)
    features = torch.FloatTensor(features)
    adj_label = np.zeros_like(adj2)
    for i in train_index:
        for j in train_index:
            if label[i] == label[j]:
                adj_label[i, j] = 1
    adj_total = torch.FloatTensor(np.eye(adj2.shape[0])) # + torch.FloatTensor(adj_corr) * torch.FloatTensor(adj_label) * adj2   # np.eye(adj2.shape[0]) # + adj_label  adj_corr  adj2
    adj = torch.FloatTensor(adj1)
    # print('adj_total.shape', adj_total.shape)
    # print('adj_total.shape', adj_total)
    model = MLP(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=label.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(args.epochs):
        loss_train, acc_train, loss_val, acc_val = train(epoch,train_index,test_index)
        max_acc = acc_val
    label_test, y_test = test(epoch,test_index)

    if label_total == []:
        label_total = label_test
        y_total = y_test
    else:
        label_total = np.hstack([label_total, label_test])
        y_total = np.vstack([y_total, y_test])

acc, pre, sen, f1, spec, kappa, auc, qwk = (all_metrics(label_total,y_total))
t_end = time.time()
t_consume = (t_end - t_start) / 60
# print( 'acc: {:.4f}'.format(acc),'auc: {:.4f}'.format(auc),'t_consume: {:.4f}'.format(t_consume))
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
        # print('Wrong_idex', i, 'age', label_age[i])
ACC1=right/label_total.shape[0]
SEN1 = right_0/(label_total.shape[0]-np.count_nonzero(label_total))
SPE1 = right_1/np.count_nonzero(label_total)
from sklearn.metrics import roc_auc_score
AUC1 = roc_auc_score(label_total, y_total[:,1])
# print('acc_sxg',ACC1,'SEN1',SEN1,'SPE1',SPE1,'AUC1',AUC1)
print('acc_sxg:', 'acc: {:.4f}'.format(ACC1),'sen: {:.4f}'.format(SEN1),'spec: {:.4f}'.format(SPE1),'auc: {:.4f}'.format(AUC1),'t_consume: {:.4f}'.format(t_consume))

# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import RandomForestClassifier
#
# X, y = load_breast_cancer(return_X_y=True)
#
# rf = RandomForestClassifier(n_estimators=100, random_state=1)
# rf.fit(features, y)
#
# importances = rf.feature_importances_
# import matplotlib.pyplot as plt
# # Plot importances
# plt.bar(range(X.shape[1]), importances)
# plt.xlabel('Feature Index')
# plt.ylabel('Feature Importance')
# plt.show()




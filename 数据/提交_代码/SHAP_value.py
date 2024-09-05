import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

all_data = pd.read_csv('combined_data2-3.csv')
all_data.info()

from sklearn.model_selection import train_test_split
train, test = train_test_split(all_data,test_size=0.2)
train_shape = train.shape
test_shape = test.shape
import lightgbm as lgb
from sklearn import metrics
params = {'objective': 'binary',
          'metric': 'binary_logloss',
          'num_round': 80,
          'verbose':1
              }#
num_round = params.pop('num_round',1000)
xtrain = lgb.Dataset(train.drop(columns=['label']), train['label'],free_raw_data=False)
xeval = lgb.Dataset(test.drop(columns=['label']), test['label'],free_raw_data=False)
evallist = [xtrain, xeval]
clf = lgb.train(params, xtrain, num_round, valid_sets=evallist)
ytrain = np.where(clf.predict(train.drop(columns=['label']))>=0.5, 1,0)
ytest = np.where(clf.predict(test.drop(columns=['label']))>=0.5, 1,0)
print("train classification report")
print(metrics.classification_report(train['label'], ytrain))
print('*'*60)
print("test classification report")
print(metrics.classification_report(test['label'], ytest))

import warnings
warnings.filterwarnings("ignore")
import shap
shap.initjs()
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(train.drop(columns=['label']))
np.array(shap_values).shape
(shap_values[0] == -1* shap_values[1]).all()
train.drop(columns=['label']).iloc[0].T
shap.initjs()
shap.plots.force(explainer.expected_value[1],shap_values[1][0],train.drop(columns=['label']).iloc[0])

explainer.expected_value
y_train_prob = clf.predict(train.drop(columns=['label']))
print('shap base value:', explainer.expected_value[1], ' log value：',np.log(y_train_prob/ (1 - y_train_prob)).mean())
shap.initjs()
plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Times New Roman' })
plt.title('MCI vs. AD',fontsize=12)
shap.summary_plot(shap_values, train.drop(columns=['label']), plot_type="bar",show=False)
fig=plt.gcf()
fig.set_size_inches(6,7.2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("mean(|SHAP value|)",fontsize=12)
plt.legend(fontsize=12)
plt.show()

#=========================#

plt.title('MCI vs. AD',fontsize=12)
shap.summary_plot(shap_values[1], train.drop(columns=['label']),show=False)
fig=plt.gcf()
fig.set_size_inches(8,7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("SHAP value",fontsize=12)
plt.show()



import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from multiprocessing import Process, Manager, Queue
manager = Manager()

import pickle

import matplotlib.pyplot as plt

from lassonet import LassoNetRegressor
from lassonet.utils import plot_path
import time
import torch

import pandas as pd
import seaborn as sbr

## Define some optimization paramaters

n_outer_folds = 5
n_inner_folds = 5

nHiddenUnits = [10,20,50,100,200,500]
nLayers = [1,2,3]
lr = [1e-4, 1e-3, 1e-2, 1e-1]

parGrid = [((x1,x2,x3),z_) for x1 in nHiddenUnits for x2 in [()]+nHiddenUnits for x3 in [()]+nHiddenUnits for z_ in lr]

parGrid = [(hiddenDims, lr) for hiddenDims, lr in parGrid if not (type(hiddenDims[2]) is not tuple and type(hiddenDims[1]) is tuple)]


def load_ctrp(drug="Lapatinib"):
    gene_exp = pd.read_csv("~/Code/Github/lassonet_exp/data/ctrpv2.gene.exp.l1k.csv")
    gene_exp = gene_exp.dropna(1)
    gene_exp.index = gene_exp.iloc[:,0]
    gene_exp = gene_exp.drop('Unnamed: 0', axis=1)
    drug_resp = pd.read_csv("~/Code/Github/lassonet_exp/data/ctrpv2.aac.csv")
    drug_resp.index = drug_resp.iloc[:,0]
    drug_resp = drug_resp.drop('Unnamed: 0', axis=1)
    drug_resp = drug_resp.loc[drug]
    drug_resp = drug_resp.dropna(0)
    unique_cells = gene_exp.columns.intersection(drug_resp.index)
    drug_resp = drug_resp[unique_cells].to_numpy()
    gene_exp = gene_exp[unique_cells].transpose().to_numpy()
    gene_exp = StandardScaler().fit_transform(gene_exp)  ## TODO: move this into CV loop
    return(gene_exp, drug_resp/100)


X, y = load_ctrp()
kf_outer = KFold(n_outer_folds, shuffle=True, random_state=42)
kf_inner = KFold(n_inner_folds, shuffle=True, random_state=5)

FullModelResDict = pickle.load(open("/Work/Code/Github/lassonet_exp/lassonet/examples/fullModelHyperparameters.p", "rb"))

bestHyperpar = np.argmin([[np.mean(x) for x in y] for items,y in list(FullModelResDict.items())], axis=0)
bestHyperpar = [list(FullModelResDict.keys())[i] for i in bestHyperpar]


splitLists = []

for train_index, test_index in kf_outer.split(X):
    train_X, train_y = X[train_index,], y[train_index]
    inner_loop_index = list(kf_inner.split(train_X))
    splitLists.append(	(train_index, test_index, inner_loop_index))



pathList = []
n_selected_outer = []
score_outer = []
lambda_outer = []
spearman_outer = []
pearson_outer = []
i=0
ii_list = []
ii = 1
for train_index, test_index, inner_loop_index in splitLists:
    valid_path_inner=[]
    train_X, train_y = X[train_index,], y[train_index]
    test_X, test_y = X[test_index,], y[test_index]
    pars=bestHyperpar[i]
    hiddenDims = tuple([x for x in pars[0] if type(x) is not tuple])
    lr = pars[1]
    model = LassoNetRegressor(eps_start=1, n_iters=(5000,5000), hidden_dims=hiddenDims, path_multiplier=1.1, lr=lr)
    for train_index, valid_index in inner_loop_index:
        inner_train_X, inner_train_y = train_X[train_index,], train_y[train_index]
        valid_X, valid_y = train_X[valid_index,], train_y[valid_index]
        scalerX = StandardScaler().fit(inner_train_X)
        # scalerY = StandardScaler().fit(inner_train_y.reshape(-1,1))
        inner_train_X = scalerX.transform(inner_train_X)
        # inner_train_y = scalerY.transform(inner_train_y.reshape(-1,1))
        valid_X = scalerX.transform(valid_X)
        # valid_y = scalerY.transform(valid_y.reshape(-1,1))
        path = model.path(inner_train_X, inner_train_y.reshape(-1), X_val=valid_X, y_val=valid_y.reshape(-1), lambda_path=np.logspace(-2,2,200))
        valid_path_inner.append(path)
        n_selected = []
        score = []
        lambda_ = []
        pearson = []
        spearman = []
        for save in path:
            model.load(save.state_dict)
            n_selected.append(save.selected.sum())
            score.append(np.sqrt(save.val_loss - save.lambda_*save.regularization))
            pearson.append(save.validation_metrics[0][0])
            spearman.append(save.validation_metrics[1][0])
            lambda_.append(save.lambda_)
        n_selected_outer.append(n_selected)
        score_outer.append(score)
        lambda_outer.append(lambda_)
        pearson_outer.append(pearson)
        spearman_outer.append(spearman)
        ii_list.append(np.repeat(ii, len(spearman)))
        ii += 1
    pathList.append(valid_path_inner)
    print(i)
    i += 1


pearson_array = np.column_stack(pearson_outer)
lambda_array = np.column_stack(lambda_outer)
score_array = np.column_stack(score_outer)
n_selected_array = np.column_stack(n_selected_outer)
spearman_array = np.column_stack(spearman_outer)
ii_array = np.column_stack(ii_list)

metricsDF = pd.DataFrame([pearson_array.flatten(), spearman_array.flatten(), score_array.flatten(), n_selected_array.flatten(), lambda_array.flatten(), ii_array.flatten()])

metricsDF = metricsDF.transpose()
metricsDF.columns = ["Pearson", "Spearman", "Score", "N_Selected", "lambda", "loop"]
metricsDF.to_csv("lapatinib_lassonet_res.csv")

#
#
#
# i=0
# pars=bestHyperpar[i]
# hiddenDims = tuple([x for x in pars[0] if type(x) is not tuple])
# lr = pars[1]
# model = LassoNetRegressor(eps_start=1, n_iters=(5000,5000), hidden_dims=hiddenDims, path_multiplier=1.5, lr=lr)
# model.load(pathList[0][3][16].state_dict)
#
# plt.plot(model.predict(X).reshape(-1), y, ".")
# plt.show()



plt.figure(figsize=(12, 12))

plt.subplot(311)
plt.grid(True)
for i in range(25):
    plt.plot(n_selected_outer[i], pearson_outer[i], ".-")
plt.xlabel("number of selected features")
plt.ylabel("Pearson")

plt.subplot(312)
plt.grid(True)
sbr.lineplot(data=metricsDF,x="lambda",y="Pearson", estimator="mean",err_style="band", ci=95)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Pearson")

plt.subplot(313)
plt.grid(True)
sbr.lineplot(data=metricsDF,x="lambda",y="N_Selected", estimator="mean",err_style="band", ci=95)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("number of selected features")
plt.show()



plt.figure(figsize=(12, 12))

plt.subplot(311)
plt.grid(True)
sbr.lineplot(data=metricsDF,x="lambda",y="Pearson", estimator="mean",err_style="band", ci=95)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Pearson")

plt.subplot(312)
plt.grid(True)
sbr.lineplot(data=metricsDF,x="lambda",y="Spearman", estimator="mean",err_style="band", ci=95)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Spearman")

plt.subplot(313)
plt.grid(True)
sbr.lineplot(data=metricsDF,x="lambda",y="Score", estimator="mean",err_style="band", ci=95)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("RMSE")
plt.show()


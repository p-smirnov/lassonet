
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
import glmnet



import pickle

import matplotlib.pyplot as plt

from lassonet import LassoNetRegressor
from lassonet.utils import plot_path
import time
import torch

import pandas as pd
import seaborn as sbr



n_outer_folds = 5
n_inner_folds = 5


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

splitLists = []

for train_index, test_index in kf_outer.split(X):
    train_X, train_y = X[train_index,], y[train_index]
    inner_loop_index = list(kf_inner.split(train_X))
    splitLists.append(	(train_index, test_index, inner_loop_index))

saveForR = []

saveForR.append(X)
saveForR.append(y)
saveForR.append(splitLists)

pickle.dump(saveForR, open("save4R.p", "wb"))

pathList = []
n_selected_outer = []
score_outer = []
lambda_outer = []
spearman_outer = []
pearson_outer = []
i=0
for train_index, test_index, inner_loop_index in splitLists:
    valid_path_inner=[]
    train_X, train_y = X[train_index,], y[train_index]
    test_X, test_y = X[test_index,], y[test_index]
    pars=bestHyperpar[i]
    hiddenDims = tuple([x for x in pars[0] if type(x) is not tuple])
    lr = pars[1]
    model = glmnet.ElasticNet(n_splits=1)
    for train_index, valid_index in inner_loop_index:
        inner_train_X, inner_train_y = train_X[train_index,], train_y[train_index]
        valid_X, valid_y = train_X[valid_index,], train_y[valid_index]
        scalerX = StandardScaler().fit(inner_train_X)
        # scalerY = StandardScaler().fit(inner_train_y.reshape(-1,1))
        inner_train_X = scalerX.transform(inner_train_X)
        # inner_train_y = scalerY.transform(inner_train_y.reshape(-1,1))
        valid_X = scalerX.transform(valid_X)
        # valid_y = scalerY.transform(valid_y.reshape(-1,1))
        path = model.fit(inner_train_X, inner_train_y.reshape(-1))
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
    pathList.append(valid_path_inner)
    print(i)
    i += 1


pearson_array = np.column_stack(pearson_outer)
lambda_array = np.column_stack(lambda_outer)
score_array = np.column_stack(score_outer)
n_selected_array = np.column_stack(n_selected_outer)
spearman_array = np.column_stack(spearman_outer)

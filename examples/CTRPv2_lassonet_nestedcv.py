#!/usr/bin/env python
# coding: utf-8

"""
Lassonet Demo Notebook - PyTorch

This notebook illustrates the Lassonet method for
feature selection on a classification task.
We will run Lassonet over [the Mice Dataset](https://archive.ics.uci.edu/ml/datasets/Mice%20Protein%20Expression).
This dataset consists of protein expression levels measured in the cortex of normal and trisomic mice who had been exposed to different experimental conditions. Each feature is the expression level of one protein.
"""
# First we import a few necessary packages


## TODO:: make number of processes lanched a parameter of script

## TODO:: fix the dependence on multiple of nprocesses


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

## Define some optimization paramaters

n_outer_folds = 5
n_inner_folds = 5

nHiddenUnits = [10, 20, 50, 100, 200, 500]
nLayers = [1, 2, 3]
lr = [1e-4, 1e-3, 1e-2, 1e-1]

parGrid = [((x1, x2, x3), z_) for x1 in nHiddenUnits for x2 in [()] + nHiddenUnits for x3 in [()] + nHiddenUnits for z_
           in lr]

parGrid = [(hiddenDims, lr) for hiddenDims, lr in parGrid if
           not (type(hiddenDims[2]) is not tuple and type(hiddenDims[1]) is tuple)]

ParameterGrid({"HU": nHiddenUnits, "depth": nLayers, "lr": lr})


def load_ctrp(drug="Lapatinib"):
    gene_exp = pd.read_csv("~/Code/Github/lassonet_exp/data/ctrpv2.gene.exp.l1k.csv")
    gene_exp = gene_exp.dropna(1)
    gene_exp.index = gene_exp.iloc[:, 0]
    gene_exp = gene_exp.drop('Unnamed: 0', axis=1)
    drug_resp = pd.read_csv("~/Code/Github/lassonet_exp/data/ctrpv2.aac.csv")
    drug_resp.index = drug_resp.iloc[:, 0]
    drug_resp = drug_resp.drop('Unnamed: 0', axis=1)
    drug_resp = drug_resp.loc[drug]
    drug_resp = drug_resp.dropna(0)
    unique_cells = gene_exp.columns.intersection(drug_resp.index)
    drug_resp = drug_resp[unique_cells].to_numpy()
    gene_exp = gene_exp[unique_cells].transpose().to_numpy()
    return (gene_exp, drug_resp / 100)


X, y = load_ctrp()
kf_outer = KFold(n_outer_folds, shuffle=True, random_state=42)
kf_inner = KFold(n_inner_folds, shuffle=True, random_state=5)

splitLists = []

for train_index, test_index in kf_outer.split(X):
    train_X, train_y = X[train_index,], y[train_index]
    inner_loop_index = list(kf_inner.split(train_X))
    splitLists.append((train_index, test_index, inner_loop_index))

t0 = time.time()
FullModelResDict = manager.dict()
time.sleep(2)


def computeForParParallel(q, resDict):
    while True:
        pars = q.get()
        if pars == -1:
            break
        hiddenDims = tuple([x for x in pars[0] if type(x) is not tuple])
        lr = pars[1]
        model = LassoNetRegressor(eps_start=1e-2, lambda_start=1, n_iters=(5000, 1000), hidden_dims=hiddenDims,
                                  path_multiplier=1.005, lr=lr)
        valid_performance_outer = []
        for train_index, test_index, inner_loop_index in splitLists:
            train_X, train_y = X[train_index,], y[train_index]
            test_X, test_y = X[test_index,], y[test_index]
            valid_performance_inner = []
            for train_index, valid_index in inner_loop_index:
                inner_train_X, inner_train_y = train_X[train_index,], train_y[train_index]
                valid_X, valid_y = train_X[valid_index,], train_y[valid_index]
                scalerX = StandardScaler().fit(inner_train_X)
                scalerY = StandardScaler().fit(inner_train_y.reshape(-1, 1))
                inner_train_X = scalerX.transform(inner_train_X)
                inner_train_y = scalerY.transform(inner_train_y.reshape(-1, 1))
                valid_X = scalerX.transform(valid_X)
                valid_y = scalerY.transform(valid_y.reshape(-1, 1))
                path = model.path(inner_train_X, inner_train_y, X_val=valid_X, y_val=valid_y, lambda_=0)
                valid_performance_inner.append(np.sqrt(path[0].val_loss - path[0].lambda_ * path[0].regularization))
            valid_performance_outer.append(valid_performance_inner)
        resDict[pars] = valid_performance_outer


q1 = Queue()
q2 = Queue()
q3 = Queue()
q4 = Queue()

p1 = Process(target=computeForParParallel, args=(q1, FullModelResDict))
p2 = Process(target=computeForParParallel, args=(q2, FullModelResDict))
p3 = Process(target=computeForParParallel, args=(q3, FullModelResDict))
p4 = Process(target=computeForParParallel, args=(q4, FullModelResDict))
p1.start()
p2.start()
p3.start()
p4.start()

queueList = [q1,q2,q3,q4]

for i in range(len(parGrid)):
    queueList[i%4].put(parGrid[i])


q1.put(-1)
q2.put(-1)
q3.put(-1)
q4.put(-1)

p1.join()
p2.join()
p3.join()
p4.join()
print(FullModelResDict)
resDict = FullModelResDict._getvalue()
t1 = time.time()

pickle.dump(resDict, open("fullModelHyperparameters.p", "wb"))
print(t1 - t0)
time.sleep(10)

# for pars in parGrid[0:10]:
# 	hiddenDims = tuple([x for x in pars[0] if type(x) is not tuple])
# 	lr = pars[1]
# 	model = LassoNetRegressor(eps_start=1e-2, lambda_start=1, n_iters=(5000,5000), hidden_dims=hiddenDims, path_multiplier=1.005, lr=lr)
# 	valid_performance_outer = []
# 	for train_index, test_index, inner_loop_index in splitLists:
# 		train_X, train_y = X[train_index,], y[train_index]
# 		test_X, test_y = X[test_index,], y[test_index]
# 		valid_performance_inner = []
# 		for train_index, valid_index in inner_loop_index:
# 			inner_train_X, inner_train_y = train_X[train_index,], train_y[train_index]
# 			valid_X, valid_y = train_X[valid_index,], train_y[valid_index]
# 			scaler = StandardScaler().fit(inner_train_X)
# 			inner_train_X = scaler.transform(inner_train_X)
# 			valid_X = scaler.transform(valid_X)
# 			path = model.path(inner_train_X, inner_train_y, X_val=valid_X, y_val=valid_y, lambda_=0)
# 			valid_performance_inner.append(np.sqrt(path[0].val_loss-path[0].lambda_*path[0].regularization))
# 		valid_performance_outer.append(valid_performance_inner)
# 	FullModelResDict[pars] = valid_performance_outer

#
# n_selected = []
# accuracy = []
#
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     y_pred = y_pred.reshape(len(y_pred))
#     accuracy.append(pearsonr(y_test, y_pred)[0])
#
# fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, "o-")
# plt.xlabel("number of selected features")
# plt.ylabel("Pearson")
# plt.title("Lapatinib")
# plt.savefig("ctrp_test.png")
# plt.show()
# plt = plot_path(model,path, X_test, y_test)
# plt.show()

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


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

from lassonet import LassoNetRegressor
from lassonet.utils import plot_path


import pandas as pd

## Define some optimization paramaters

n_outer_folds = 5
n_inner_folds = 5

nHiddenUnits = [10,20,50,100,200,500]
nLayers = [1,2,3]




def load_ctrp(drug="Lapatinib"):
	gene_exp = pd.read_csv("../data/ctrpv2.gene.exp.l1k.csv")
	gene_exp = gene_exp.dropna(1)
	gene_exp.index = gene_exp.iloc[:,0]
	gene_exp = gene_exp.drop('Unnamed: 0', axis=1)
	drug_resp = pd.read_csv("../data/ctrpv2.aac.csv")
	drug_resp.index = drug_resp.iloc[:,0]
	drug_resp = drug_resp.drop('Unnamed: 0', axis=1)
	drug_resp = drug_resp.loc[drug]
	drug_resp = drug_resp.dropna(0)
	unique_cells = gene_exp.columns.intersection(drug_resp.index)
	drug_resp = drug_resp[unique_cells].to_numpy()
	gene_exp = gene_exp[unique_cells].transpose().to_numpy()
	gene_exp = StandardScaler().fit_transform(gene_exp)  ## TODO: move this into CV loop
	return(gene_exp, drug_resp)


X, y = load_ctrp()
kf_outer = KFold(n_outer_folds)
kf_inner = KFold(n_inner_folds)


for train_index, test_index in kf.split(X):
	train_X, train_y = X[train_index,], y[train_index]
	test_X, test_y = X[test_index,], y[test_index]
	for train_index, valid_index in kf.split(X):
		train_X, train_y = train_X[train_index,], train_y[train_index]
		valid_X, valid_y = train_X[valid_index,], train_y[valid_index]
		model = LassoNetRegressor(eps_start=1e-2, lambda_start=1, n_iters=(5000,5000), hidden_dims=(100,), path_multiplier=1.005)
		path = model.path(X_train, y_train)



n_selected = []
accuracy = []

for save in path:
    model.load(save.state_dict)
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(len(y_pred))
    accuracy.append(pearsonr(y_test, y_pred)[0])

fig = plt.figure(figsize=(9, 6))
plt.grid(True)
plt.plot(n_selected, accuracy, "o-")
plt.xlabel("number of selected features")
plt.ylabel("Pearson")
plt.title("Lapatinib")
# plt.savefig("ctrp_test.png")
plt.show()
plt = plot_path(model,path, X_test, y_test)
plt.show()

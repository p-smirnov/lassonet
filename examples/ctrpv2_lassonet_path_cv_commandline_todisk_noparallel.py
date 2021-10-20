import os
import gc
scratchPath = os.getenv('SCRATCH')

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
import datetime
manager = Manager()

import pickle

import matplotlib.pyplot as plt

from lassonet import LassoNetRegressor
from lassonet.utils import plot_path
import time
import torch

import pandas as pd
import seaborn as sbr

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("drug", help="Drug name for which to run the script")
args = parser.parse_args()

drugName = args.drug

print(datetime.datetime.now())
## takes about 14 hours

## Define some optimization paramaters

n_folds = 5
run_prefix=scratchPath + "/full_batch_grid_search_M"
batch_size=None

nHiddenUnits = [10,20,50,100,200,500]
nLayers = [1,2,3]
lr = [1e-4, 1e-3, 1e-2, 1e-1]

parGrid = [((x1,x2,x3),z_) for x1 in nHiddenUnits for x2 in [()]+nHiddenUnits for x3 in [()]+nHiddenUnits for z_ in lr]

parGrid = [(hiddenDims, lr) for hiddenDims, lr in parGrid if not (type(hiddenDims[2]) is not tuple and type(hiddenDims[1]) is tuple)]


def returnCVResults(bestHyperpar, eps_start, n_iters,
                   backtrack, verbose, lambda_seq, patience,
                   batch_size, X, y, M, splitLists):
    pathList = []
    best_list = []

    for train_index, valid_index in splitLists:
        train_X, train_y = X[train_index,], y[train_index]
        valid_X, valid_y = X[valid_index,], y[valid_index]
        pars=bestHyperpar
        hiddenDims = tuple([x for x in pars[0] if type(x) is not tuple])
        lr = pars[1]

        scalerX = StandardScaler().fit(train_X)
        # scalerY = StandardScaler().fit(inner_train_y.reshape(-1,1))
        train_X = scalerX.transform(train_X)
        # inner_train_y = scalerY.transform(inner_train_y.reshape(-1,1))
        valid_X = scalerX.transform(valid_X)
        model = LassoNetRegressor(eps_start=eps_start, n_iters=n_iters, hidden_dims=hiddenDims,
                              backtrack=backtrack, verbose=verbose, lr=lr, lambda_seq=lambda_seq, patience=patience,
                              batch_size=batch_size, M=M)
        path = model.path(train_X, train_y.reshape(-1), X_val=valid_X, y_val=valid_y.reshape(-1))
        best_tuple = model._return_best_performance(path)
        pathList.append(path)
        best_list.append(best_tuple)
    best_val_loss_ave = np.mean([x[1] for x in best_list])
    return(model, pathList, best_list, best_val_loss_ave)






def patternSearchM(bestHyperpar, eps_start, n_iters,
                   backtrack, verbose, lambda_seq, patience,
                   batch_size, X, y, splitLists, M_start=10, min_step=0.1, start_step=5):
    (cur_model, cur_path, cur_best_list, cur_best_ave_val_loss) = returnCVResults(bestHyperpar=bestHyperpar,
                                                                                  eps_start=eps_start,
                                                                                  n_iters=n_iters,
                                                                                  backtrack=backtrack,
                                                                                  verbose=verbose,
                                                                                  lambda_seq=lambda_seq,
                                                                                  patience=patience,
                                                                                  batch_size=batch_size,
                                                                                  X=X, y=y,
                                                                                  M=M_start, splitLists=splitLists)

    cur_M = M_start
    cur_step = start_step


    while(cur_step>=min_step):
        M_forward = cur_M+cur_step
        M_backward = cur_M-cur_step
        (for_model, for_path, for_best_list, for_best_ave_val_loss) = returnCVResults(bestHyperpar=bestHyperpar,
                                                                                  eps_start=eps_start,
                                                                                  n_iters=n_iters,
                                                                                  backtrack=backtrack,
                                                                                  verbose=verbose,
                                                                                  lambda_seq=lambda_seq,
                                                                                  patience=patience,
                                                                                  batch_size=batch_size,
                                                                                  X=X, y=y,
                                                                                  M=M_forward, splitLists=splitLists)
        (back_model, back_path, back_best_list, back_best_ave_val_loss) = returnCVResults(bestHyperpar=bestHyperpar,
                                                                                  eps_start=eps_start,
                                                                                  n_iters=n_iters,
                                                                                  backtrack=backtrack,
                                                                                  verbose=verbose,
                                                                                  lambda_seq=lambda_seq,
                                                                                  patience=patience,
                                                                                  batch_size=batch_size,
                                                                                  X=X, y=y,
                                                                                  M=M_backward, splitLists=splitLists)
        direction = np.argmin((for_best_ave_val_loss, back_best_ave_val_loss))
        new_best = (for_best_ave_val_loss,back_best_ave_val_loss)[direction]
        if(new_best<cur_best_ave_val_loss):
            cur_best_ave_val_loss=new_best
            cur_best_list = (for_best_list, back_best_list)[direction]
            cur_M = (M_forward, M_backward)[direction]
            cur_path = (for_path, back_path)[direction]
            cur_model = (for_model, back_model)[direction]
            print(f"Current M is {cur_M}")
        else:
            cur_step=cur_step/2
            print(f"Current step reduced to {cur_step}")

    return (cur_best_list, cur_M, cur_model, cur_path)


def dumpCurrentModel(arg1, arg2, arg3, arg4):
    pickle.dump((arg1,arg2,arg3,arg4), open(run_prefix + "/" + "best_model_"+drugName+".p", "wb"))

def gridSearchM(bestHyperpar, eps_start, n_iters,
                   backtrack, verbose, lambda_seq, patience,
                   batch_size, X, y, splitLists, M_grid=np.logspace(-1,3,10)):
    cur_best_ave_val_loss = np.inf

    for M in M_grid:

        (new_model, new_path, new_best_list, new_best_ave_val_loss) = returnCVResults(bestHyperpar=bestHyperpar,
                                                                                  eps_start=eps_start,
                                                                                  n_iters=n_iters,
                                                                                  backtrack=backtrack,
                                                                                  verbose=verbose,
                                                                                  lambda_seq=lambda_seq,
                                                                                  patience=patience,
                                                                                  batch_size=batch_size,
                                                                                  X=X, y=y,
                                                                                  M=M, splitLists=splitLists)

        if(new_best_ave_val_loss<cur_best_ave_val_loss):
            cur_best_ave_val_loss=new_best_ave_val_loss
            cur_best_list = new_best_list
            cur_M = M
            cur_path = new_path
            cur_model = new_model
            print(f"Current M is {cur_M}")
        gc.collect()

    return (cur_best_list, cur_M, cur_model, cur_path)


def gridSearchMSaveDisk(bestHyperpar, eps_start, n_iters,
                   backtrack, verbose, lambda_seq, patience,
                   batch_size, X, y, splitLists, M_grid=np.logspace(-1,3,10)):
    cur_best_ave_val_loss = np.inf
    firstLoop=True
    for M in M_grid:
        (new_model, new_path, new_best_list, new_best_ave_val_loss) = returnCVResults(bestHyperpar=bestHyperpar,
                                                                                  eps_start=eps_start,
                                                                                  n_iters=n_iters,
                                                                                  backtrack=backtrack,
                                                                                  verbose=verbose,
                                                                                  lambda_seq=lambda_seq,
                                                                                  patience=patience,
                                                                                  batch_size=batch_size,
                                                                                  X=X, y=y,
                                                                                  M=M, splitLists=splitLists)


        if(new_best_ave_val_loss<cur_best_ave_val_loss):
            dumpCurrentModel(new_best_list, M, new_model, new_path)
            cur_best_ave_val_loss=new_best_ave_val_loss
            cur_M = M
            print(f"Current M is {cur_M}")
        del new_path
        gc.collect()

    (cur_best_list, cur_M, cur_model, cur_path) = pickle.load(open(run_prefix + "/" + "best_model_"+drugName+".p", "rb"))
    return (cur_best_list, cur_M, cur_model, cur_path)



def load_ctrp(drug):
    gene_exp = pd.read_csv("examples/data/ctrpv2.gene.exp.l1k.csv")
    gene_exp = gene_exp.dropna(1)
    gene_exp.index = gene_exp.iloc[:,0]
    gene_exp = gene_exp.drop('Unnamed: 0', axis=1)
    drug_resp = pd.read_csv("examples/data/ctrpv2.aac.csv")
    drug_resp.index = drug_resp.iloc[:,0]
    drug_resp = drug_resp.drop('Unnamed: 0', axis=1)
    drug_resp = drug_resp.loc[drug]
    drug_resp = drug_resp.dropna(0)
    unique_cells = gene_exp.columns.intersection(drug_resp.index)
    drug_resp = drug_resp[unique_cells].to_numpy()
    gene_exp = gene_exp[unique_cells].transpose().to_numpy()
    return(gene_exp, drug_resp/100)

def load_ctrp2(drug):
    data = pd.read_csv("examples/L1000_gene_expression/gene_CCLE_rnaseq_" + drug + "_response.csv")
    drug_resp = data.target
    gene_exp = data.drop('cell_line', axis=1).drop('target', axis=1)
    return(gene_exp, drug_resp)




print(drugName)
X, y = load_ctrp(drugName)
kf_outer = KFold(n_folds, shuffle=True, random_state=42)


FullModelResDict = pickle.load(open(run_prefix + "/fullModelHyperparameters_"+drugName+".p", "rb"))

hyperparResTable = np.array([[np.mean(x) for x in y] for items,y in list(FullModelResDict.items())])
bestHyperparIndex = np.argmin(np.mean(hyperparResTable, 1))
bestHyperpar = list(FullModelResDict.keys())[bestHyperparIndex]



splitLists = []

for train_index, valid_index in kf_outer.split(X):
    splitLists.append((train_index, valid_index))
forRGLMNET = [X, y, splitLists]
pickle.dump(forRGLMNET, open(run_prefix + "/" + "forR_"+drugName+".p", "wb"))


(best_res_over_l_list, best_M, best_model, pathList) = gridSearchMSaveDisk(bestHyperpar=bestHyperpar, eps_start=1,
                                                                      n_iters=(5000,5000),
                                                                      backtrack=True, verbose=False,
                                                                      lambda_seq=np.logspace(-2,2,2000),
                                                                      patience=(100,100),
                                                                      batch_size=batch_size, X=X, y=y,
                                                                      splitLists=splitLists)


print(best_M)

pickle.dump((best_res_over_l_list, best_M, best_model, pathList), open(run_prefix + "/" + "best_model_"+drugName+".p", "wb"))

n_selected_outer = []
score_outer = []
lambda_outer = []
spearman_outer = []
pearson_outer = []
i=0
ii_list = []
best_M_list = []
ii = 1
for best_path in pathList:
    n_selected = []
    score = []
    lambda_ = []
    pearson = []
    spearman = []
    for save in best_path:
        best_model.load(save.state_dict)
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
    best_M_list.append(np.repeat(best_M, len(spearman)))
    ii += 1


pearson_array = np.concatenate(pearson_outer)
lambda_array = np.concatenate(lambda_outer)
score_array = np.concatenate(score_outer)
n_selected_array = np.concatenate(n_selected_outer)
spearman_array = np.concatenate(spearman_outer)
ii_array = np.concatenate(ii_list)
M_array = np.concatenate(best_M_list)

metricsDF = pd.DataFrame([pearson_array.flatten(), spearman_array.flatten(), score_array.flatten(),
                          n_selected_array.flatten(), lambda_array.flatten(), ii_array.flatten(), M_array.flatten()])

metricsDF = metricsDF.transpose()
metricsDF.columns = ["Pearson", "Spearman", "Score", "N_Selected", "lambda", "loop", "Chosen_M"]
metricsDF.to_csv(run_prefix + "/" + drugName + "_lassonet_res.csv")

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
plt.title(drugName)
plt.subplot(311)
plt.grid(True)
for i in range(5):
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
plt.savefig(run_prefix + "/" +drugName+"_plot1.png")



plt.figure(figsize=(12, 12))
plt.title(drugName)
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
plt.savefig(run_prefix + "/" +drugName+"_plot2.png")

print(datetime.datetime.now())

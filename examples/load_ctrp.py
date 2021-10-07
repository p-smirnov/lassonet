import numpy as np
import pandas as pd


def load_ctrp(drug):
    gene_exp = pd.read_csv("data/ctrpv2.gene.exp.l1k.csv")
    gene_exp = gene_exp.dropna(1)
    gene_exp.index = gene_exp.iloc[:,0]
    gene_exp = gene_exp.drop('Unnamed: 0', axis=1)
    drug_resp = pd.read_csv("data/ctrpv2.aac.csv")
    drug_resp.index = drug_resp.iloc[:,0]
    drug_resp = drug_resp.drop('Unnamed: 0', axis=1)
    drug_resp = drug_resp.loc[drug]
    drug_resp = drug_resp.dropna(0)
    unique_cells = gene_exp.columns.intersection(drug_resp.index)
    drug_resp = drug_resp[unique_cells].to_numpy()
    gene_exp = gene_exp[unique_cells].transpose().to_numpy()
    return(gene_exp, drug_resp/100)

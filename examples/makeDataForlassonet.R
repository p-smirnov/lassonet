library(PharmacoGx)
library(data.table)
CCLE.CTRPv2 <- readRDS('~/Data/TBPInputs/rna/CCLE.CTRPv2.rds')


rna.seq <- SummarizedExperiment::assay(
	summarizeMolecularProfiles(CCLE.CTRPv2, "Kallisto_0.46.1.rnaseq"))


l1k_genes <- fread("geneinfo_beta.txt")

rna.seq <- rna.seq[gsub(rownames(rna.seq), pat="\\.[0-9]+$", rep="") %in% l1k_genes[feature_space=="landmark",ensembl_id],]

aac <- summarizeSensitivityProfiles(CCLE.CTRPv2, "aac_recomputed")

write.csv(rna.seq, file="ctrpv2.gene.exp.l1k.csv")

write.csv(aac, file="ctrpv2.aac.csv")


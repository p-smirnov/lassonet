library(reticulate)
library(glmnet)

source_python("pickleHelper.py")


drugNames=c("5-Fluorouracil", "AZD7762", "AZD8055", "Bortezomib", "Crizotinib", "Dabrafenib", "Dasatinib", "Docetaxel", "Erlotinib", "Gefitinib", "Gemcitabine", "Ibrutinib", "JQ1 compound", "Lapatinib", "MK-2206", "Nilotinib", "Paclitaxel", "Pictilisib", "PLX4720", "Vincristine", "Vorinostat")
drugNames=c("5-Fluorouracil", "Lapatinib")



toPlotDrug <- list()
lassonetDrug <- list()
lassonetDrugSearchM <- list()

for(drug in drugNames){


	data <- read_pickle_file(paste0("examples/full_batch_backtrack/forR_",drug,".p"))



	X <- data[[1]]
	y <- data[[2]]

	splitIndicies <- data[[3]]
	ii <- 1
	dataList =list()
	topGeneCor <- numeric(length(splitIndicies))

	for(i in seq_along(splitIndicies)){

		train_outer_index <- splitIndicies[[i]][[1]]
		valid_index <- splitIndicies[[i]][[2]]

		train_X <- X[train_outer_index,]
		train_y <- y[train_outer_index]

		valid_X <- X[valid_index,]
		valid_y <- y[valid_index]


	
		model = glmnet(train_X, train_y, nlambda=200, lambda.min.ratio=1e-7)		
		
		top_gene <- which.max(abs(cor(train_X, train_y)))

		predict_y = predict(model, valid_X)

		topGeneCor[[i]] = abs(cor(valid_X[,top_gene], valid_y))

		dataList[[i]]  = data.frame("Pearson"=cor(predict_y, valid_y),"Lambda"=model$lambda, nSelected= model$df, loop=as.character(i))


	}



	library(ggplot2)
	library(data.table)

	toPlot = rbindlist(dataList)
	lassonet_res <- fread(paste0("examples/full_batch_backtrack/",drug,"_lassonet_res.csv"))

	lassonet_res_searchM <- fread(paste0("examples/full_batch_search_M/",drug,"_lassonet_res.csv"))


	toPlotDrug[[drug]] <- toPlot
	lassonetDrug[[drug]] <- lassonet_res
	lassonetDrugSearchM[[drug]] <- lassonet_res_searchM


	# pdf(paste0(drug,"_pearson_over_nselect_points.pdf"))
	# p <- ggplot() + geom_point(data=toPlot, aes(nSelected, Pearson, col="glmnet")) + 
	# geom_point(data=lassonet_res, aes(x=N_Selected, y=Pearson, col="Lassonet")) + theme_bw() +
	# geom_hline(mapping=aes(col="Top Gene", yintercept= mean(topGeneCor))) + ggtitle(paste0("Results for ",drug," in 5FCV"))
	# print(p)	
	# dev.off()

	# pdf(paste0(drug,"_pearson_over_lambda_points.pdf"))
	# p <- ggplot() + geom_point(data=toPlot, aes(Lambda, Pearson, col="glmnet")) + scale_x_log10() + 
	# geom_point(data=lassonet_res, aes(x=lambda, y=Pearson, col="Lassonet")) + theme_bw() +
	# geom_hline(mapping=aes(col="Top Gene", yintercept= mean(topGeneCor))) + ggtitle(paste0("Results for ",drug," in 5FCV"))
	# print(p)	
	# dev.off()

	# pdf(paste0(drug,"_pearson_over_nselect_smooth.pdf"))
	# p <- ggplot() + geom_smooth(data=toPlot, aes(nSelected, Pearson, col="glmnet")) + 
	# geom_smooth(data=lassonet_res, aes(x=N_Selected, y=Pearson, col="Lassonet")) + theme_bw() +
	# geom_hline(mapping=aes(col="Top Gene", yintercept= mean(topGeneCor))) + ggtitle(paste0("Results for ",drug," in 5FCV"))
	# print(p)	
	# dev.off()

	# pdf(paste0(drug,"_pearson_over_lambda_smooth.pdf"))
	# p <- ggplot() + geom_smooth(data=toPlot, aes(Lambda, Pearson, col="glmnet")) + scale_x_log10() + 
	# geom_smooth(data=lassonet_res, aes(x=lambda, y=Pearson, col="Lassonet")) + theme_bw() +
	# geom_hline(mapping=aes(col="Top Gene", yintercept= mean(topGeneCor))) + ggtitle(paste0("Results for ",drug," in 5FCV"))
	# print(p)
	# dev.off()


	# res.summary <- data.frame(lassonet=lassonet_res[,max(Pearson,na.rm=T), loop][,V1], glmnet=toPlot[,max(Pearson, na.rm=T), loop][,V1],
	# 			"Top Gene"=topGeneCor)

	# pdf(paste0(drug,"_res_summary.pdf"))
	# boxplot(res.summary, main=drug, ylab="Best pearson correlation over path in validation")
	# dev.off()



}


for(drug in names(lassonetDrug)){
	lassonetDrug[[drug]][,Drug:=drug]
}

lassonetAllDrug <- rbindlist(lassonetDrug)

for(drug in names(lassonetDrugSearchM)){
	lassonetDrugSearchM[[drug]][,Drug:=drug]
}

lassonetAllDrugSearchM <- rbindlist(lassonetDrugSearchM)



for(drug in names(toPlotDrug)){
	toPlotDrug[[drug]][,Drug:=drug]
}

toPlotAllDrug <- rbindlist(toPlotDrug)


toPlotAllDrugBest <-  toPlotAllDrug[,max(Pearson,na.rm=T), .(loop,Drug)]

lassonetAllDrugBest <-  lassonetAllDrug[,max(Pearson,na.rm=T), .(loop,Drug)]
lassonetAllDrugBestSearchM <-  lassonetAllDrugSearchM[,max(Pearson,na.rm=T), .(loop,Drug)]



toPlotAllDrugBest[,Method:="glmnet"]
lassonetAllDrugBest[,Method:="lassonet"]
lassonetAllDrugBestSearchM[,Method:="lassonet Search M"]


toPlotTotal <- rbind(rbind(toPlotAllDrugBest,lassonetAllDrugBest), lassonetAllDrugBestSearchM)

colnames(toPlotTotal)[3]<- "Pearson"

toPlotTotal <- toPlotTotal[,.(Mean=mean(Pearson), upper=mean(Pearson)+ sd(Pearson), lower=mean(Pearson)-sd(Pearson)),.(Drug,Method)]

pdf("full_batch_backtrack_lassonet_vs_glmnet_summary_searchM.pdf", height=6, width=9)
ggplot(toPlotTotal, aes(Drug, Mean, fill=Method)) + 
geom_col(position="dodge") +
geom_errorbar(aes(ymin=lower, ymax=upper), width=.2,
                 position=position_dodge(.9)) +
theme_bw() + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1)) + ylab("Mean Pearson")
dev.off()
# 6/21 lassonet wins






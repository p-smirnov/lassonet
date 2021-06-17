library(reticulate)
library(glmnet)

source_python("pickleHelper.py")

data <- read_pickle_file("save4R.p")



X <- data[[1]]
y <- data[[2]]

splitIndicies <- data[[3]]
ii <- 1
dataList =list()
topGeneCor <- numeric(25)

for(i in seq_along(splitIndicies)){

	train_outer_index <- splitIndicies[[i]][[1]]
	test_index <- splitIndicies[[i]][[2]]
	splitInner <- splitIndicies[[i]][[3]]

	train_X <- X[train_outer_index,]
	train_y <- y[train_outer_index]

	test_X <- X[test_index,]
	test_y <- y[test_index]


	for(j in seq_along(splitInner)){

		train_inner_index <- splitInner[[j]][[1]]
		valid_index <- splitInner[[j]][[2]]
	
		train_inner_X <- train_X[train_inner_index,]
		train_inner_y <- train_y[train_inner_index]
		valid_X <- train_X[valid_index,]
		valid_y <- train_y[valid_index]

		model = glmnet(train_inner_X, train_inner_y, nlambda=200, lambda.min.ratio=1e-7)		
		
		top_gene <- which.max(cor(train_inner_X, train_inner_y))

		predict_y = predict(model, valid_X)

		topGeneCor[[ii]] = cor(valid_X[,595], valid_y)

		dataList[[ii]]  = data.frame("Pearson"=cor(predict_y, valid_y),"Lambda"=model$lambda, nSelected= model$df, loop=as.character(ii))

		ii = ii + 1

	}

}



library(ggplot2)
library(data.table)

toPlot = rbindlist(dataList)
lassonet_res <- fread("examples/lapatinib_lassonet_res.csv")



pdf("lap_pearson_over_nselect_points.pdf")
ggplot() + geom_point(data=toPlot, aes(nSelected, Pearson, col="glmnet")) + 
geom_point(data=lassonet_res, aes(x=N_Selected, y=Pearson, col="Lassonet")) + theme_bw() +
geom_hline(mapping=aes(col="Top Gene", yintercept= mean(topGeneCor))) + ggtitle("Results for Lapatinib in nested CV")
dev.off()

pdf("lap_pearson_over_lambda_points.pdf")
ggplot() + geom_point(data=toPlot, aes(Lambda, Pearson, col="glmnet")) + scale_x_log10() + 
geom_point(data=lassonet_res, aes(x=lambda, y=Pearson, col="Lassonet")) + theme_bw() +
geom_hline(mapping=aes(col="Top Gene", yintercept= mean(topGeneCor))) + ggtitle("Results for Lapatinib in nested CV")
dev.off()

pdf("lap_pearson_over_nselect_smooth.pdf")
ggplot() + geom_smooth(data=toPlot, aes(nSelected, Pearson, col="glmnet")) + 
geom_smooth(data=lassonet_res, aes(x=N_Selected, y=Pearson, col="Lassonet")) + theme_bw() +
geom_hline(mapping=aes(col="Top Gene", yintercept= mean(topGeneCor))) + ggtitle("Results for Lapatinib in nested CV")
dev.off()

pdf("lap_pearson_over_lambda_smooth.pdf")
ggplot() + geom_smooth(data=toPlot, aes(Lambda, Pearson, col="glmnet")) + scale_x_log10() + 
geom_smooth(data=lassonet_res, aes(x=lambda, y=Pearson, col="Lassonet")) + theme_bw() +
geom_hline(mapping=aes(col="Top Gene", yintercept= mean(topGeneCor))) + ggtitle("Results for Lapatinib in nested CV")
dev.off()


res.summary <- data.frame(lassonet=lassonet_res[,max(Pearson,na.rm=T), loop][,V1], glmnet=toPlot[,max(Pearson, na.rm=T), loop][,V1],
			"Top Gene"=topGeneCor)

pdf("lapatinib_res_summary.pdf")
boxplot(res.summary)
dev.off()



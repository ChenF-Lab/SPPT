############# Binary classifiers with selected features ################
# example : SPPT vs SNPT

## 1. Random forest
library(caret)
pos_neg_dataTrain_filtered<-read.csv("SPPT_SNPT_features_selected-trainSet.csv",header=T)
pos_neg_dataTest_filtered<-read.csv("SPPT_SNPT_features_selected-testSet.csv",header=T)

fitControl <- trainControl(method = "repeatedcv", number =10, repeats = 1,
                           returnResamp = "all")
tuneGrid <- expand.grid(mtry=c(3,5,10,20,50,100,300,500,700,900,1000))
set.seed(666)
rfFit1 <- train(y~.,pos_neg_dataTrain_filtered,method = "rf",metric="Accuracy",
                trControl = fitControl,tuneGrid = tuneGrid,verbose = FALSE)


## confusion Matrix
pos_neg_dataTest_filtered_pred<-predict(rfFit1,pos_neg_dataTest_filtered)
confusionMatrix(pos_neg_dataTest_filtered_pred,pos_neg_dataTest_filtered$y,
                            positive="pos")

## ROC
library(pROC) 
pos_vs_neg_roc <- roc(pos_neg_dataTest_filtered$y,as.numeric(pos_neg_dataTest_filtered_pred))
p<-plot(pos_vs_neg_roc, print.auc=TRUE, auc.polygon=T,col = "steelblue",print.auc.col = "black",
         max.auc.polygon=F, auc.polygon.col="aliceblue",print.thres=F,main='pos vs neg ROC',
         cex.axis=1.8,cex.lab=2,cex.main=2,mgp = c(3, 1, 0),mar = c(5, 5, 4, 2), grid=TRUE,
         identity.lty=2,print.auc.cex=1.5)

## 2. Support vector machine

library(e1071)
set.seed(666)
tuned<-tune.svm(y~.,data = pos_neg_dataTrain_filtered,gamma = 10^(-6:-1),cost = 10^(1:2))
model.tuned<-svm(y~.,data = pos_neg_dataTrain_filtered,gamma=tuned$best.parameters$gamma,cost=tuned$best.parameters$cost,cross=10)
svm.tuned.pred<-predict(model.tuned,pos_neg_dataTest_filtered)
svm.tuned.table<-table(svm.tuned.pred,pos_neg_dataTest_filtered$y)
confusionMatrix(svm.tuned.table,mode = "everything",positive="pos")
## 10次交叉验证
accuracy_pos_con<-model.tuned$accuracies


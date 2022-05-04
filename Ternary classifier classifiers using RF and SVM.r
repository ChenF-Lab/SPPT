############# Ternary classifiers with selected features ################
#  SPPT vs SNPT vs Ctrl
library(caret)

## 1. Random forest
pos_neg_nonTB_dataTrain_filtered<-read.csv("SPPT_SNPT_Ctrl_features_selected-trainSet.csv",header=T)
pos_neg_nonTB_dataTest_filtered<-read.csv("SPPT_SNPT_Ctrl_features_selected-testSet.csv",header=T)

fitControl <- trainControl(method = "repeatedcv", number =10, repeats = 10,
tuneGrid <- expand.grid(mtry=c(3,5,10,20,50,100,300,500,700,900,1000))
set.seed(666)
rfFit1 <- train(y~.,pos_neg_nonTB_dataTrain_filtered,method = "rf",metric="Accuracy",
                trControl = fitControl,tuneGrid = tuneGrid,verbose = FALSE)

pos_neg_nonTB_dataTest_filtered_pred<-predict(rfFit1,pos_neg_nonTB_dataTest_filtered)
confusionMatrix(pos_neg_nonTB_dataTest_filtered_pred,pos_neg_nonTB_dataTest_filtered$y,
                            positive="pos")

## 2. Support vector machine
library(e1071)
set.seed(666)
tuned<-tune.svm(y~.,data = pos_neg_nonTB_dataTrain_filtered,gamma = 10^(-6:-1),cost = 10^(1:2))
model.tuned<-svm(y~.,data = pos_neg_nonTB_dataTrain_filtered,gamma=tuned$best.parameters$gamma,cost=tuned$best.parameters$cost,cross=10)
svm.tuned.pred<-predict(model.tuned,pos_neg_nonTB_dataTest_filtered)
svm.tuned.table<-table(svm.tuned.pred,pos_neg_nonTB_dataTest_filtered$y)
confusionMatrix(svm.tuned.table,mode = "everything")
## 10次交叉验证
accuracy_pos_neg_con<-model.tuned$accuracies


############# Data preprocessing and feature selection: Code availability ################
## 1. Reading input file 
df<-read.csv("feature_120_na_delDrug.csv")
df$y<-factor(df$y)

## 2. Spliting training set and validation set
library(caret)
set.seed(666)
trainIndex<-createDataPartition(df$y,p=0.8,list=FALSE,times=1) ## training vs validation set = 8:2
dataTrain<-df[trainIndex,]
dataTest<-df[-trainIndex,]

## 3. Imputing missing values
imputation_k<-preProcess(dataTrain,method='knnImpute')
dataTrain_nona<-predict(imputation_k,dataTrain)
dataTest_nona<-predict(imputation_k,dataTest)

## 4. correlation analysis
dataTrain_nona_cor<-cor(dataTrain_nona[,-1])
highlyCorfeature<-findCorrelation(dataTrain_nona_cor,cutoff=0.7,names=T,verbose=F)
highlyCorfeature_index<-findCorrelation(dataTrain_nona_cor,cutoff=0.7,names=F,verbose=F)
dataTrain_nocor<-dataTrain_nona[,-(highlyCorfeature_index+1)]
dim(dataTrain_nocor)

## 5. Feature selection

## 1. SPPT vs SNPT
pos_neg_dataTrain<-dataTrain_nocor[dataTrain_nocor$y=="pos"|dataTrain_nocor$y=="neg",]
pos_neg_dataTrain$y<-factor(pos_neg_dataTrain$y)
pos_neg_dataTest<-dataTest_nona[dataTest_nona$y=="pos"|dataTest_nona$y=="neg",]
pos_neg_dataTest$y<-factor(pos_neg_dataTest$y)

set.seed(666)
ctrl<- rfeControl(functions = rfFuncs, method = "cv",verbose = FALSE, returnResamp = "final")
pos_neg_featureselector<-rfe(pos_neg_dataTrain[,-1],pos_neg_dataTrain[,1],
       sizes=1:(ncol(pos_neg_dataTrain)-1),rfeControl=ctrl)
pos_neg_featureselector_result<-pos_neg_featureselector$results
pos_neg_featureimportance<-pos_neg_featureselector$fit$importance
pos_neg_featureimportance<-pos_neg_featureimportance[order(pos_neg_featureimportance[,4],decreasing=T),]

pos_neg_dataTrain_filtered<-pos_neg_dataTrain[,c("y",rownames(pos_neg_featureimportance)[1:3])]
pos_neg_dataTest_filtered<-pos_neg_dataTest[,c("y",rownames(pos_neg_featureimportance)[1:3])]

write.csv(pos_neg_dataTrain_filtered,"SPPT_SNPT_features_selected-trainSet.csv",rownames=F)
write.csv(pos_neg_dataTest_filtered,"SPPT_SNPT_features_selected-testSet.csv",rownames=F)

##############################################################################

##2. SPPT vs Ctrl
pos_con_dataTrain<-dataTrain_nocor[dataTrain_nocor$y=="pos"|dataTrain_nocor$y=="con",]
pos_con_dataTrain$y<-factor(pos_con_dataTrain$y)
pos_con_dataTest<-dataTest_nona[dataTest_nona$y=="pos"|dataTest_nona$y=="con",]
pos_con_dataTest$y<-factor(pos_con_dataTest$y)

set.seed(666)
ctrl<- rfeControl(functions = rfFuncs, method = "cv",verbose = FALSE, returnResamp = "final")
pos_con_featureselector<-rfe(pos_con_dataTrain[,-1],pos_con_dataTrain[,1],
       sizes=1:(ncol(pos_con_dataTrain)),rfeControl=ctrl)
pos_con_featureselector_result<-pos_con_featureselector$results
pos_con_featureimportance<-pos_con_featureselector$fit$importance
pos_con_featureimportance<-pos_con_featureimportance[order(pos_con_featureimportance[,4],decreasing=T),]

pos_con_dataTrain_filtered<-pos_con_dataTrain[,c("y",rownames(pos_con_featureimportance)[1:2])]
pos_con_dataTest_filtered<-pos_con_dataTest[,c("y",rownames(pos_con_featureimportance)[1:2])]

write.csv(pos_con_dataTrain_filtered,"SPPT_Ctrl_features_selected-trainSet.csv",rownames=F)
write.csv(pos_con_dataTest_filtered,"SPPT_Ctrl_features_selected-testSet.csv",rownames=F)

##############################################################################

## 3. SNPT vs Ctrl
neg_con_dataTrain<-dataTrain_nocor[dataTrain_nocor$y=="neg"|dataTrain_nocor$y=="con",]
neg_con_dataTrain$y<-factor(neg_con_dataTrain$y)
neg_con_dataTest<-dataTest_nona[dataTest_nona$y=="neg"|dataTest_nona$y=="con",]
neg_con_dataTest$y<-factor(neg_con_dataTest$y)

set.seed(666)
ctrl<- rfeControl(functions = rfFuncs, method = "cv",verbose = FALSE, returnResamp = "final")
neg_con_featureselector<-rfe(neg_con_dataTrain[,-1],neg_con_dataTrain[,1],
       sizes=1:(ncol(neg_con_dataTrain)-1),rfeControl=ctrl)
neg_con_featureselector_result<-neg_con_featureselector$results
neg_con_featureimportance<-neg_con_featureselector$fit$importance
neg_con_featureimportance<-neg_con_featureimportance[order(neg_con_featureimportance[,4],decreasing=T),]

neg_con_dataTrain_filtered<-neg_con_dataTrain[,c("y",rownames(neg_con_featureimportance)[1:3])]
neg_con_dataTest_filtered<-neg_con_dataTest[,c("y",rownames(neg_con_featureimportance)[1:3])]

write.csv(neg_con_dataTrain_filtered,"SNPT_Ctrl_features_selected-trainSet.csv",rownames=F)
write.csv(neg_con_dataTest_filtered,"SNPT_Ctrl_features_selected-testSet.csv",rownames=F)

##############################################################################

## 4. SPPT vs SNPT vs Ctrl

set.seed(666)
ctrl<- rfeControl(functions = rfFuncs, method = "cv",verbose = FALSE, returnResamp = "final")
pos_neg_nonTB_featureselector<-rfe(dataTrain_nocor[,-1],dataTrain_nocor[,1],
       sizes=1:(ncol(dataTrain_nocor)-1),rfeControl=ctrl)
pos_neg_nonTB_featureselector_result<-pos_neg_nonTB_featureselector$results
pos_neg_nonTB_featureimportance<-pos_neg_nonTB_featureselector$fit$importance
pos_neg_nonTB_featureimportance<-pos_neg_nonTB_featureimportance[order(pos_neg_nonTB_featureimportance[,4],decreasing=T),]

pos_neg_nonTB_dataTrain_filtered<-dataTrain_nocor[,c("y",rownames(pos_neg_nonTB_featureimportance)[1:9])]
pos_neg_nonTB_dataTest_filtered<-dataTest_nona[,c("y",rownames(pos_neg_nonTB_featureimportance)[1:9])]

write.csv(pos_neg_nonTB_dataTrain_filtered,"SPPT_SNPT_Ctrl_features_selected-trainSet.csv",rownames=F)
write.csv(pos_neg_nonTB_dataTest_filtered,"SPPT_SNPT_Ctrl_features_selected-testSet.csv",rownames=F)



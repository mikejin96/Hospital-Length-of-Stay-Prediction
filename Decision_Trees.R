# Libraries
library(caret)
library(caTools)
library(dplyr)
library(randomForest)
library(gbm) 
library(rpart)
library(rpart.plot) 
library(tibble)
library(DiagrammeR)
library(knitr)

# Data Cleaning
data=read.csv("Model_Data_V3_ohc.csv")
data=na.omit(data)
data = data[,2:52]
glimpse(data)

# Training/Testing
set.seed(1)
train=sample(1:nrow(data),nrow(data)/2)
data.train=data[train,]
data.test=data[-train,]

# CART Tree cross validation
cp_val <- data.frame(.cp = c(0.000001,.00001,0.0001,0.001,0.01,0.1))

cp_cv = train(TOTAL_LOS~.,
             trControl=trainControl(method="cv",number=10), data=data.train, method="rpart",minbucket=5,
             tuneGrid=cp_val, metric="Rsquared", maximize=TRUE)

# Best cp using cross validation
cp_best = cp_cv$bestTune
cp_best

# Plotting R^2 using CV vs cp value
ggplot(cp_cv$results, aes(x=factor(cp), y=Rsquared)) +
  geom_col() +
  theme_bw() +
  ylim(0,0.5) +
  xlab("cp value") +
  ylab("R^2 using Cross Validation") +
  theme(axis.title=element_text(size=18), axis.text=element_text(size=14))

# Building the cross-validated tree
tree.cv = rpart(TOTAL_LOS ~ ., 
                data=data.train, 
                minbucket = 5,
                cp=cp_best)

# Small tree
tree.small = rpart(TOTAL_LOS ~ ., 
                   data=data.train, 
                   minbucket = 5,
                   cp=0.005)

# Image of tree.cv
prp(tree.cv,digits = 2, varlen = 0, faclen = 0, tweak = 1)

# Image of tree.small
tree_plot=prp(tree.small,digits = 2, varlen = 0, faclen = 0, tweak = 1)

# Random forests OOB
rf_oob <- train(x = data.train %>% select(-TOTAL_LOS),
                      y = data.train$TOTAL_LOS,
                      method="rf",
                      tuneGrid=data.frame(mtry=1:12),  
                      ntree=100,
                      nodesize=10,
                      trControl=trainControl(method="oob"))

# Best mtry using OOB
mtry_best <- train.rf.oob$bestTune[[1]]
mtry_best

# Final RF model
rf_model = randomForest(TOTAL_LOS~., data=data.train, ntree=500,nodesize=10,mtry=rf_oob$bestTune[[1]])

# Variables that play the most important role in the final model
rf_imp <- data.frame(imp=importance(rf_model))

rf_imp_name=rownames_to_column(rf_imp, var="Name_of_Variable")

imp_var = rf_imp_name %>%
  arrange(desc(IncNodePurity)) %>% 
  head(10) 

Variable_Names=c("Injury", "Parasitic_Disease", "Digestive_System", "respiratory_system",
                 "Age", "Weight", "Ill_Defined_Condition", "Blood_Disease", "Skin_Disease", 
                 "Metabolic_Disease")

imp_var[,1]=Variable_Names

# Plotting variable importance
var_imp_plot=ggplot(data=imp_var, aes(x = reorder(Name_of_Variable, IncNodePurity), IncNodePurity), hor) +
  geom_bar(stat="identity") +
  xlab("Variable Name") +
  ggtitle("Variable Importance Using Random Forests")+
  theme(axis.text=element_text(size=16),
        axis.title=element_text(size=16,face="bold")) +
  coord_flip()

var_imp_plot

# Boosting (manual calibration)
boost_model = gbm(TOTAL_LOS~.,data=data.train,distribution = "gaussian",n.minobsinnode = 1, n.trees=5000, shrinkage=0.05, interaction.depth=10)

var_imp1_boosting = summary(boost_model)

# Variables that play the most important role in boosting
top_10_imp1= var_imp1_boosting %>%
  arrange(desc(rel.inf)) %>%
  head(10)

Variable_Names_boost=c("Age", "Parasitic_Disease", "Injury", "Weight",
                       "Digestive_System", "Respiratory_System", "Ill_Defined_Condition", "Metabolic_Disease", "Blood_Disease", 
                       "Circulatory_System")

top_10_imp1[,1]=Variable_Names_boost

# Plotting variable importance
var_imp_plot_boost=ggplot(data=top_10_imp1, aes(x = reorder(var, rel.inf), rel.inf), hor) +
  geom_bar(stat="identity") +
  xlab("Variable Name") +
  ggtitle("Variable Importance Using Boosting")+
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold")) +
  coord_flip()

var_imp_plot_boost

# Key Metrics

# Cross-validated Tree

# in-sample R^2
pred.tree.cv.train = predict(tree.cv, newdata = data.train)

SSE.tree.cv.train <- sum((pred.tree.cv.train - data.train$TOTAL_LOS)^2)

SST.tree.cv.train <- sum((mean(data.train$TOTAL_LOS) - data.train$TOTAL_LOS)^2)

tree.cv.train.OSR <- 1 - SSE.tree.cv.train/SST.tree.cv.train

tree.cv.train.OSR

# in-sample MAE
MAE.tree.cv.train=mean(abs(pred.tree.cv.train-data.train$TOTAL_LOS))
MAE.tree.cv.train

# in-sample RMSE
RMSE.tree.cv.train=sqrt(mean((pred.tree.cv.train-data.train$TOTAL_LOS)^2))
RMSE.tree.cv.train

# out-of-sample R^2
pred.tree.cv.test = predict(tree.cv, newdata = data.test)

SSE.tree.cv.test <- sum((pred.tree.cv.test - data.test$TOTAL_LOS)^2)

SST.tree.cv.test <- sum((mean(data.test$TOTAL_LOS) - data.test$TOTAL_LOS)^2)

tree.cv.test.OSR <- 1 - SSE.tree.cv.test/SST.tree.cv.test

tree.cv.test.OSR

# out-of-sample MAE
MAE.tree.cv.test=mean(abs(pred.tree.cv.test-data.test$TOTAL_LOS))
MAE.tree.cv.test

# out-of-sample RMSE
RMSE.tree.cv.test=sqrt(mean((pred.tree.cv.test-data.test$TOTAL_LOS)^2))
RMSE.tree.cv.test

# Random Forest

# in-sample R^2
pred.rf.train = predict(rf_model, newdata = data.train)

SSE.rf.train <- sum((pred.rf.train - data.train$TOTAL_LOS)^2)

SST.rf.train <- sum((mean(data.train$TOTAL_LOS) - data.train$TOTAL_LOS)^2)

rf.train.OSR <- 1 - SSE.rf.train/SST.rf.train

rf.train.OSR

# in-sample MAE
MAE.rf.train=mean(abs(pred.rf.train-data.train$TOTAL_LOS))
MAE.rf.train

# in-sample RMSE
RMSE.rf.train=sqrt(mean((pred.rf.train-data.train$TOTAL_LOS)^2))
RMSE.rf.train

# out-of-sample R^2
pred.rf.test = predict(rf_model, newdata = data.test)

SSE.rf.test <- sum((pred.rf.test - data.test$TOTAL_LOS)^2)

SST.rf.test <- sum((mean(data.test$TOTAL_LOS) - data.test$TOTAL_LOS)^2)

rf.test.OSR <- 1 - SSE.rf.test/SST.rf.test

rf.test.OSR

# out-of-sample MAE
MAE.rf.test=mean(abs(pred.rf.test-data.test$TOTAL_LOS))
MAE.rf.test

# out-of-sample RMSE
RMSE.rf.test=sqrt(mean((pred.rf.test-data.test$TOTAL_LOS)^2))
RMSE.rf.test

# Boosting Model

# in-sample R^2
pred.boost1.train = predict(boost_model , newdata = data.train, n.trees=5000)

SSE.boost1.train <- sum((pred.boost1.train - data.train$TOTAL_LOS)^2)

SST.boost1.train <- sum((mean(data.train$TOTAL_LOS) - data.train$TOTAL_LOS)^2)

boost1.train.OSR <- 1 - SSE.boost1.train/SST.boost1.train

boost1.train.OSR

# in-sample MAE
MAE.boost1.train=mean(abs(pred.boost1.train-data.train$TOTAL_LOS))
MAE.boost1.train

# in-sample RMSE
RMSE.boost1.train=sqrt(mean((pred.boost1.train-data.train$TOTAL_LOS)^2))
RMSE.boost1.train

# out-of-sample R^2
pred.boost1.test = predict(boost_model, newdata = data.test, n.trees=5000)

SSE.boost1.test <- sum((pred.boost1.test - data.test$TOTAL_LOS)^2)

SST.boost1.test <- sum((mean(data.test$TOTAL_LOS) - data.test$TOTAL_LOS)^2)

boost1.test.OSR <- 1 - SSE.boost1.test/SST.boost1.test

boost1.test.OSR

# out-of-sample MAE
MAE.boost1.test=mean(abs(pred.boost1.test-data.test$TOTAL_LOS))
MAE.boost1.test

# out-of-sample RMSE
RMSE.boost1.test=sqrt(mean((pred.boost1.test-data.test$TOTAL_LOS)^2))
RMSE.boost1.test

# ORT results
OCT.train.OSR=0.4117618
MAE.OCT.train=5.4380806
RMSE.OCT.train=8.4447225
OCT.test.OSR=0.3051417
MAE.OCT.test=5.5653004
RMSE.OCT.test=8.9326954

# Summary
summary_table <- data.frame(
  Model = c (" Model 1 (CART)", "Model 2 (OCT)",
             "Model 3 (Random Forests)", "Model 4 (Boosting)"),
  in_sample_R2= c(tree.cv.train.OSR, OCT.train.OSR, 
                  rf.train.OSR, boost1.train.OSR),
  in_sample_MAE= c(MAE.tree.cv.train, MAE.OCT.train, 
                   MAE.rf.train, MAE.boost1.train),
  in_sample_RMSE= c(RMSE.tree.cv.train, RMSE.OCT.train,
                    RMSE.rf.train, RMSE.boost1.train),
  out_sample_R2= c(tree.cv.test.OSR, OCT.test.OSR,
                   rf.test.OSR, boost1.test.OSR),
  out_sample_MAE= c(MAE.tree.cv.test, MAE.OCT.test,
                    MAE.rf.test, MAE.boost1.test),
  out_sample_RMSE= c(RMSE.tree.cv.test, RMSE.OCT.test, 
                     RMSE.rf.test, RMSE.boost1.test))

# Printing the summary table		
print(summary_table) 

# Exporting predicted LOS (using Boosting) and actual LOS on test set
write.csv(data.test, "pred_actual_LOS.csv")









if (!require("tidyverse")) install.packages("tidyverse")
if (!require("BiocManager"))    install.packages("BiocManager")
if (!require("EBImage"))   BiocManager::install("EBImage")
if (!require("caret"))   install.packages("caret")
if (!require("doParallel"))   install.packages("doParallel")
if (!require("e1071"))   install.packages("e1071")
if (!require("naivebayes"))   install.packages("naivebayes")
if (!require("glmnet"))   install.packages("glmnet")
if (!require("kernlab"))   install.packages("kernlab")
if (!require("Matrix"))   install.packages("Matrix")
if (!require("xgboost"))   install.packages("xgboost")
if (!require("randomForest"))   install.packages("randomForest")

library(EBImage)
library(tidyverse)
library(caret)
library(doParallel)
library(e1071)
library(naivebayes)
library(glmnet)
library(kernlab)
library(Matrix)
library(xgboost)
library(randomForest)

# Load the preprocessed data
load("./data/preprocesseddata_resnet18.Rda")

#Shuffle the training set to remove any sorting in data
set.seed(1)
sample_idx <- sample(1:nrow(train_x))
train_x <- train_x[sample_idx,]
train_y <- data.frame(label = as.character(train_y$label[sample_idx]))

# Register the cores to do parallel processing
cluster <- makeCluster(detectCores() -1)  
registerDoParallel(cluster)


# Naive Bayes
set.seed(1)
# Training controls
trControl <- trainControl(method = "repeatedcv", number = 10, repeats =3, allowParallel = TRUE)
# Train the model
train_nb <- train(train_x, train_y$label,
                  method = "naive_bayes",trControl= trControl)
# Predict the validation set
pred_nb_valid <- predict(train_nb, valid_x)
# Predict the test set
pred_nb_test <- predict(train_nb, test_x)

# save the model
save(train_nb,pred_nb_valid,pred_nb_test,file="./data/nb_model.Rda")

# Stop the parallel processing cluster
stopCluster(cluster)
registerDoSEQ()

# Start the cluster again
cluster <- makeCluster(detectCores() -1)  
registerDoParallel(cluster)


# Logistic regression
# Training controls
set.seed(1)
trControl <- trainControl(method="repeatedcv", number=10, repeats = 3,allowParallel = TRUE)
tuneGrid <- expand.grid(alpha = 1,
                        lambda = c(5e-5,1e-4,5e-4,1e-3,5e-3))
# Train the model
train_lr <- train(train_x, train_y$label,
                  method = "glmnet",family = "multinomial",
                  type.multinomial = "grouped",
                  trControl = trControl,tuneGrid = tuneGrid )
# Plot the training parameters
plot(train_lr)
# Predict the validation set
pred_lr_valid <- predict(train_lr, valid_x)
# Predict the test set
pred_lr_test <- predict(train_lr, test_x)

# save the model
save(train_lr,pred_lr_test,pred_lr_valid,file="./data/lr_model.Rda")

# Stop the parallel processing cluster
stopCluster(cluster)
registerDoSEQ()

# Start the cluster again
cluster <- makeCluster(detectCores() -1)  
registerDoParallel(cluster)


# Support Vector machine with guassian kernel
# training parameters
set.seed(1)
# Training controls
trControl <- trainControl(method="repeatedcv", number=10, repeats = 3,allowParallel = TRUE)
# Train the model
train_svm <- train(train_x, train_y$label,
                   method = "svmRadial",trControl = trControl, 
                   tuneLength = 10)
# Plot the training parameters
plot(train_svm)
# Predict the validation set
pred_svm_valid <- predict(train_svm, valid_x)
# Predict the test set
pred_svm_test <- predict(train_svm, test_x)

# save the model
save(train_svm,pred_svm_valid,pred_svm_test,file="./data/svm_model.Rda")

# Stop the parallel processing cluster
stopCluster(cluster)
registerDoSEQ()

# Start the cluster again
cluster <- makeCluster(detectCores() -1)  
registerDoParallel(cluster)


# Random forest
# training parameters
set.seed(1)
# Training controls
trControl <- trainControl(method="repeatedcv", number=10,repeats = 3,allowParallel = TRUE)
tuneGrid <- expand.grid(mtry = c(11,17,23,29,35))
# Train the model
train_rf <- train(train_x, train_y$label,
                  method = "parRF",ntree = 1000,
                  trControl = trControl,tuneGrid =tuneGrid,
                  importance = TRUE)
# Plot the training parameters
plot(train_rf)
# Predict the validation set
pred_rf_valid <- predict(train_rf, valid_x)
# Predict the test set
pred_rf_test <- predict(train_rf, test_x)

# save the model
save(train_rf,pred_rf_valid,pred_rf_test,file="./data/rf_model.Rda")

# Stop the parallel processing cluster
stopCluster(cluster)
registerDoSEQ()

# Start the cluster again
cluster <- makeCluster(detectCores() -1)  
registerDoParallel(cluster)



# XGBoost forest
# training parameters
set.seed(1)
# Training controls
trControl <- trainControl(method="repeatedcv", number=10,repeats = 3,allowParallel = TRUE)
tuneGrid <- expand.grid(colsample_bytree=.8,nrounds = c(100,200,300,400,500),eta = c(.2,.3,.4),
                        gamma = 0,subsample = .8, max_depth = 1,min_child_weight = 1)
# Train the model
train_xgb <- train(train_x, train_y$label,
                   method = "xgbTree", 
                   trControl = trControl,objective = "multi:softprob",
                   tuneGrid = tuneGrid,
                   numclass = 10)
# Plot the training parameters
plot(train_xgb)
# Predict the validation set
pred_xgb_valid <- predict(train_xgb, valid_x)

# Predict the test set
pred_xgb_test <- predict(train_xgb, test_x)

pred_xgb_test <- predict(train_xgb, test_x)

# save the model
save(train_xgb,pred_xgb_test,pred_xgb_valid,file="./data/xgb_model.Rda")

# Stop the parallel processing cluster
stopCluster(cluster)
registerDoSEQ()

# Loading all the models
load("./data/cnn_model.Rda")
load("./data/nb_model.Rda")
load("./data/lr_model.Rda")
load("./data/svm_model.Rda")
load("./data/rf_model.Rda")
load("./data/xgb_model.Rda")


# Calculate the validation accuracy of models
acc_cnn_valid <- mean(pred_cnn_valid == valid_y$label)
acc_nb_valid <- mean(pred_nb_valid == valid_y$label)
acc_lr_valid <- mean(pred_lr_valid == valid_y$label)
acc_svm_valid <- mean(pred_svm_valid == valid_y$label)
acc_rf_valid <- mean(pred_rf_valid == valid_y$label)
acc_xgb_valid <- mean(pred_xgb_valid == valid_y$label)

# Ensemble model
# Data frame with predictions of validation dataset from different models
ens_df <- cbind(pred_cnn_valid,pred_nb_valid,pred_lr_valid,pred_svm_valid,
                pred_rf_valid,pred_xgb_valid)
# Assign the maximally predicted class in the ensemble
pred_ens_valid <- sapply(seq(1,nrow(ens_df)),function(idx){
  # Frequency of elements in a row
  tt <- table(ens_df[idx,])
  # Select the first maximum if there is a tie
  names(tt[tt==max(tt)])[1]
}) 
# Convert to factor
pred_ens_valid <- factor(as.numeric(pred_ens_valid),labels = levels(pred_nb_valid))
# Validation accuracy 
acc_ens_valid <- mean(pred_ens_valid == valid_y$label)

acc <- data.frame(method=c("CNN Resnet 18","Naive Bayes","Logistic regression",
                           "SVM with Radial","Random Forest",
                           "XGBoost","Ensemble"),
                  accuracy = c(acc_cnn_valid,acc_nb_valid,acc_lr_valid,acc_svm_valid,
                               acc_rf_valid,acc_xgb_valid,acc_ens_valid))

acc %>% knitr::kable(.)

# Final test set accuracy
# Ensemble model
# Data frame with predictions of test set from different models
ens_df <- cbind(pred_cnn_test,pred_nb_test,pred_lr_test,pred_svm_test,
                pred_rf_test,pred_xgb_test)

ens_df <- cbind(pred_lr_test,pred_svm_test)
# Assign the maximally predicted class in the ensemble
pred_ens_test <- sapply(seq(1,nrow(ens_df)),function(idx){
  # Frequency of elements in a row
  tt <- table(ens_df[idx,])
  # Select the first maximum if there is a tie
  names(tt[tt==max(tt)])[1]
}) 
# Convert to factor
pred_ens_test <- factor(as.numeric(pred_ens_test),labels = levels(pred_nb_test))
# Test accuracy
acc_ens_test <- mean(pred_ens_test == test_y$label)

acc <- data.frame(method="Ensemble",
                  accuracy = acc_ens_test)

acc %>% knitr::kable(.)

# Confusion matrix of final model
cm <- confusionMatrix(pred_ens_test,test_y$label)
cm <- as_tibble(cm$table)

# Plot the confusion matrix
ggplot(cm, aes(x=Prediction, y= reorder(Reference, desc(Reference)) , fill=n)) +
  geom_tile() + theme_bw() + coord_equal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))  +
  scale_fill_distiller(palette="Blues", direction=1) +
  guides(fill=F) + # removing legend for `fill`
  labs(title = "Confusion Matrix") + # using a title 
  geom_text(aes(label=n), color="black") +# printing values 
  ylab('Reference')


# index of Picasso paintings misclassified as Matisse
idx <- seq(1,500)[(pred_ens_test == 'henri.matisse' & test_y$label == 'pablo.picasso')]
# corresponding paths in images
filepaths <- test_generator$filepaths[idx]

par(mfrow = c(3, 2),oma=c(0,0,2,0))
for (i in filepaths){
  plot(readImage(i))
}
title("Picasso paintings misclassified as Matisse", outer=TRUE) 
par(mfrow = c(1, 1))

# index of Picasso paintings misclassified as Van gogh
idx <- seq(1,500)[(pred_ens_test == 'vincent.van.gogh' & test_y$label == 'pablo.picasso')]
# corresponding paths in images
filepaths <- test_generator$filepaths[idx]

par(mfrow = c(2, 2),oma=c(0,0,2,0))
for (i in filepaths){
  plot(readImage(i))
}
title("Picasso paintings misclassified as Van gogh", outer=TRUE) 
par(mfrow = c(1, 1))

# index of Van gogh  paintings misclassified as Matisse
idx <- seq(1,500)[(pred_ens_test == 'claude.monet' & test_y$label == 'vincent.van.gogh')]
# corresponding paths in images
filepaths <- test_generator$filepaths[idx]

par(mfrow = c(2, 2),oma=c(0,0,2,0))
for (i in filepaths){
  plot(readImage(i))
}
title("Van gogh paintings misclassified as Monet", outer=TRUE) 
par(mfrow = c(1, 1))
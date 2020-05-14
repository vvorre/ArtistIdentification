if (!require("keras"))   install.packages("keras")
if (!require("Rfast"))   install.packages("Rfast")
if (!require("caret"))   install.packages("caret")
if (!require("tidyverse")) install.packages("tidyverse")


# Make sure EBimage is detached as it conflicts with keras
detach("package:EBImage", unload=TRUE)

library(Rfast)
library(tidyverse)
library(caret)
library(keras)
# Install keras for neural network training, Need to have Anaconda installed as a prerequisite
install_keras()

# Image dimensions and batchsize to be used for processing cnn
img_width <- 224
img_height <- 224
batchsize <- 100

# Paths to directory
train_dir <- "./data/procdata/train/"
validation_dir <- "./data/procdata/validation/"
test_dir <- "./data/procdata/test/"

# Generators for train, test and validation datasets with feature normalization
# Data augmentation using horizontal flip and zoom for train
train_datagen <- image_data_generator(
  featurewise_center = TRUE,
  featurewise_std_normalization = TRUE,
  zoom_range = 0.2, # zoom by +_ .2
  horizontal_flip = TRUE, # horizontal flip
  fill_mode = "nearest"
)
validation_datagen <- image_data_generator(
  featurewise_center = TRUE,
  featurewise_std_normalization = TRUE
)
test_datagen <- image_data_generator(
  featurewise_center = TRUE,
  featurewise_std_normalization = TRUE
)
# Controls to read images from the train, test and validation directories 
train_generator <- flow_images_from_directory(
  train_dir,                  # Target directory  
  train_datagen,              # Data generator
  target_size = c(img_width, img_height),  # Size is 224 × 224
  batch_size = batchsize,
  class_mode = "categorical", # categorical labels
  shuffle = TRUE ,           # Shuffle the classes
  seed = 1
)
validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(img_width, img_height),
  batch_size = batchsize,
  shuffle = FALSE,
  class_mode = "categorical"
)
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(img_width, img_height),
  batch_size = batchsize,
  shuffle = FALSE,
  class_mode = "categorical"
)

# Number of samples in each set
train_samples <- train_generator$n
validation_samples <- validation_generator$n
test_samples <- test_generator$n

## Load Resnet 18- downloaded from https://github.com/qubvel/classification_models
base_model <- load_model_hdf5(filepath = './data/resnet18_224.h5')
# model on top of the base model
model <- keras_model_sequential() %>% 
  base_model %>% 
  layer_global_average_pooling_2d() %>% # global averaging
  layer_dense(units = 10, activation = "softmax") # 10 layer node

# Freeze weights for all the weights of the model and the global averaging layer
freeze_weights(base_model)
for (layer in base_model$layers)
  layer$trainable <- FALSE
for (layer in model$layers[2])
  layer$trainable <- FALSE

# Compile the model with the adam optimizer
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999),
  metrics = c("accuracy")
)

# Callbacks to the model to store and earstop the model, if val accuraccy doesn't improve
callbacks <- list(
  callback_early_stopping(monitor = "val_acc", min_delta = .005,# Early stopping if the validation accuracy
                          patience = 3, restore_best_weights = TRUE), # does not improve
  callback_model_checkpoint(filepath = paste0("./data/","resnet18_coarsetune",
                                              "weights.{epoch:02d}-{val_acc:.3f}.hd5"),# Save weights of every run
                            monitor = "val_acc"))
# Store the history
history_coarse <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = round(train_samples/batchsize),
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = round(validation_samples/batchsize),
  callbacks = callbacks
)
# Save the history and best model
save(history_coarse,file = './data/resnet18_coarse_hist.Rda')
save_model_hdf5(model,'./data/resnet18_coarse_model.hd5', overwrite = TRUE)

# Fine tuning the model
# Unfreeze all the layers
unfreeze_weights(base_model)
for (layer in base_model$layers)
  layer$trainable <- TRUE
for (layer in model$layers[2])
  layer$trainable <- TRUE

# Callbacks
callbacks <- list(
  callback_early_stopping(monitor = "val_acc", min_delta = .005,# Early stopping if the validation accuracy
                          patience = 3, restore_best_weights = TRUE), # does not improve
  callback_model_checkpoint(filepath = paste0("./data/","resnet18_finetune",
                                              "weights.{epoch:02d}-{val_acc:.3f}.hd5"),# Save weights of every run
                            monitor = "val_acc"))
# Compile the model with reduced learning rate
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(lr = 1e-4, beta_1 = 0.9, beta_2 = 0.999),
  metrics = c("accuracy")
)
# Train the model and store the history
history_fine <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = round(train_samples/batchsize),
  epochs = 20,
  validation_data = validation_generator,
  validation_steps = round(validation_samples/batchsize),
  initial_epoch = 10, callbacks = callbacks
)

save(history_fine,file = './data/resnet18_fine_hist.Rda')
save_model_hdf5(model,'./data/resnet18_fine_model.hd5', overwrite = TRUE)

# load the history data
load(file = './data/resnet18_coarse_hist.Rda')
load(file = './data/resnet18_fine_hist.Rda')
# Data frame with combined metrics from coarse and fine tuning
acc_df <- data.frame(n = seq(1:20), train_acc = c(history_coarse$metrics$acc,history_fine$metrics$acc),
                        val_acc = c(history_coarse$metrics$val_acc,history_fine$metrics$val_acc))
# Make it to long format
acc_df <- gather(acc_df, key,value ,-n)
# Plot the data
acc_df %>%
  ggplot(aes(x = n,y = value, group_by(key),color = key)) +
  geom_line()+
  geom_point()+ 
  xlab("Iterations") +
  ylab("Accuracy") + 
  ggtitle("Validation and Training accuracy for Resnet 18")

# Load the model
model <- load_model_hdf5(filepath = './data/resnet18_fine_model.hd5')

# Validation set predictions
# Predict prob of classes
predict_prob_valid <- predict_generator(model, 
                                        validation_generator, 
                                        steps = round(validation_samples/batchsize))
# Predict the class with max prob
predict_class_valid <- rowMaxs(predict_prob_valid, value = FALSE) - 1

# Predict the validation set by assigning the factor variables
pred_cnn_valid <- factor(predict_class_valid,
                         labels = colnames(data.frame((validation_generator$class_indices))))
# Test set predictions
# Predict prob of classes
predict_prob_test <- predict_generator(model, 
                                       test_generator, 
                                       steps = round(test_samples/batchsize))
# Predict the class with max prob
predict_class_test <- rowMaxs(predict_prob_test, value = FALSE) - 1

# Predict the test set by assigning the factor variables
pred_cnn_test <- factor(predict_class_test,
                        labels = colnames(data.frame((test_generator$class_indices))))

# Save the values
save(pred_cnn_valid,pred_cnn_test,file="./data/cnn_model.Rda")


# Extract feature vectors
# Generators for train without any augmentation
train_datagen <- image_data_generator(
  featurewise_center = TRUE,
  featurewise_std_normalization = TRUE
)
train_generator <- flow_images_from_directory(
  train_dir, # Target directory
  train_datagen, # Data generator with preprocessing
  target_size = c(img_width, img_height), # input image size
  batch_size = batchsize, # batch size
  shuffle = FALSE, # if the data needs to be shuffled
  class_mode = "categorical"
)
# Extract the model with the output layer as the global averaging 512 elements 
model_extract <- keras_model(inputs = model$input, 
                             outputs = get_layer(model, 'global_average_pooling2d_1')$output)

# Extract the features by using the model
test_x <- as.data.frame(predict_generator(model_extract, 
                                          test_generator, 
                                          steps = round(test_samples/batchsize)))

valid_x <- as.data.frame(predict_generator(model_extract, 
                                           validation_generator, 
                                           steps = round(validation_samples/batchsize)))

train_x <- as.data.frame(predict_generator(model_extract, 
                                           train_generator, 
                                           steps = round(train_samples/batchsize)))
# Extract the corresponding labels
test_y <- data.frame(label = factor(test_generator$classes,
                                    labels = colnames(data.frame((test_generator$class_indices)))))

valid_y <- data.frame(label = factor(validation_generator$classes,
                                     labels = colnames(data.frame((validation_generator$class_indices)))))

train_y <- data.frame(label = factor(train_generator$classes,
                                     labels = colnames(data.frame((train_generator$class_indices)))))

# Save the data to a file
save(train_x,train_y,valid_x,valid_y,test_x,test_y,file="./data/preprocesseddata_resnet18.Rda")


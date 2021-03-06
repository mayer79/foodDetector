#======================================================================
# Use EPFL data and pretrain GoogLeNet to detect food
#======================================================================

#======================================================================
# Required packages
#======================================================================

library(mxnet)
library(imager)
library(abind)
library(ranger)
library(glmnet)
library(lightgbm)

#======================================================================
# Some functions
#======================================================================

source("r/functions.R")

#======================================================================
# Load and prepare the pretrained Inception V3 googlenet
#======================================================================

model <- mx.model.load("Inception/Inception-BN", iteration = 126)

# Modify the model to output the last features before going to softmax output
internals <- model$symbol$get.internals()
fea_symbol <- internals[[match("global_pool_output", internals$outputs)]]
temp <- model
temp$arg.params$fc1_bias <- NULL
temp$arg.params$fc1_weight <- NULL

model2 <- list(symbol = fea_symbol,
               arg.params = temp$arg.params,
               aux.params = temp$aux.params)

class(model2) <- "MXFeedForwardModel"

# Load in the mean image, which is used for preprocessing using:
mean.img <- as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])

# Labels
labels <- as.vector(read.delim("Inception/synset.txt", header = FALSE)$V1)

#======================================================================
# Prepare image data sets and predict deep feature vector
#======================================================================
# 
# train <- preproc.images("data/training", center = mean.img)
# train$X <- t(adrop(predict(model2, X = train$X), 1:2))
# gc()
# valid <- preproc.images("data/validation", center = mean.img)
# valid$X <- t(adrop(predict(model2, X = valid$X), 1:2))
# gc()
# test <- preproc.images("data/evaluation", center = mean.img)
# test$X <- t(adrop(predict(model2, X = test$X), 1:2))
# gc()
# 
# save(train, valid, test, file = "data/food_pretrained.RData")

load("data/food_pretrained.RData", verbose = TRUE)

#======================================================================
# PCA and logistic regression
#======================================================================

# PCA to reduce dimension to m
set.seed(67754)
pr <- princomp(train$X, cor = TRUE)

# Find number of PCs via CV
for (i in 2:60) {
  print(i)
  fit_glmnet <- cv.glmnet(x = pr$scores[, 1:i, drop = FALSE], 
                          y = factor(train$y), 
                          family = "binomial", 
                          lambda = c(0, 0.1), 
                          alpha = 0,
                          nfolds = 5)
  cat("cvm: ", fit_glmnet$cvm[2])
}

m <- 29 # Optimum value by CV
dat <- data.frame(y = factor(train$y), pr$scores[, seq_len(m), drop = FALSE])

fit_lr <- glm(y ~ ., data = dat, family = "binomial")
pred <- predict(fit_lr, dat, type = "response")
print(mean(round(pred) != train$y)) # 0.013

# Predict validation data
valid$pcaScores <- as.data.frame(predict(pr, valid$X)[, seq_len(m)])
pred <- round(predict(fit_lr, data.frame(valid$pcaScores), type = "response"))
mean(pred != valid$y) # 0.016


#======================================================================
# Penalized logistic regression
#======================================================================

# Tune elastic net parameters on validation data
set.seed(3345)
for (a in seq(0, 1, by = 0.1)) {
  print(a)
  fit_glmnet <- cv.glmnet(x = train$X,
                          y = factor(train$y), 
                          family = "binomial", 
                          alpha = a, 
                          nfolds = 5)
  cat("cvm: ", min(fit_glmnet$cvm))
  cat("lambda: ", fit_glmnet$lambda.min, "\n")
}

# Optimal values seem to be alpha 0.1, lambda = 0.003426268
set.seed(399)
fit_glmnet <- glmnet(x = train$X, 
                     y = factor(train$y), 
                     family = "binomial", 
                     alpha = 0.1, 
                     lambda = 0.003426268)
pred <- predict(fit_glmnet, valid$X, type = "class") 
print(mean(pred != valid$y)) # 0.01

save(fit_glmnet, file = "data/food_glmnet.RData")

#======================================================================
# Random forest 
#======================================================================

# OOB optimized mtry
set.seed(234924)
dat <- data.frame(y = factor(train$y), train$X)
for (i in seq(1, 45, by = 1)) {
  print(i)
  fit_rf <- ranger(y ~ ., data = dat, mtry = i, num.trees = 100)
  print(fit_rf$prediction.error) # 1.5% OOB
}

# Optimium seems to be mtry = 24
fit_rf <- ranger(y ~ ., 
                 data = dat, 
                 mtry = 24, 
                 seed = 4384, 
                 importance = "impurity")
fit_rf # 1.6 % 
sort(importance(fit_rf))

# Predict validation data
pred <- predict(fit_rf, data = data.frame(valid$X))$predictions
mean(pred != valid$y) # 0.014

#======================================================================
# Stack of lightGBM models
#======================================================================

set.seed(45332)
dtrain <- lgb.Dataset(data = train$X, 
                      label = train$y)

paramGrid <- expand.grid(iteration = NA_integer_, # filled by algorithm
                         score = NA_real_,     # "
                         learning_rate = c(0.05, 0.06, 0.07), # (0:10)/10 -> use c(0.05, 0.06, 0.07)
                         max_depth = 3:4,  # 2:7 -> 3:4                       
                         num_leaves = c(63, 127), #  c(31, 63, 127) -> use 63, 127               
                         min_data_in_leaf = c(10, 40), # c(1, 10, 40) -> c(10, 40)
                         min_gain_to_split = 0, #(0:5)/10 -> 0
                         lambda_l1 = 0.01, #c(0, 0.001, 0.01, 0.1, 1), # use 0.01 or 0.05
                         lambda_l2 = 0.05, #c(0, 0.001, 0.01, 0.1, 1), # use 0.05 or 0.01 
                         feature_fraction = (5:10)/10, #(5:10)/10 -> all are good,
                         bagging_fraction = 0.7, #(5:10)/10 -> 0.5 to 0.8 are good, 0.7 seems best
                         bagging_freq = 8,                         
                         max_bin = 255, #c(63, 127, 255),-> use 255
                         nthread = 7)

(n <- nrow(paramGrid)) # 48

for (i in seq_len(n)) {
  gc(verbose = FALSE) # clean memory
  
  cvm <- lgb.cv(as.list(paramGrid[i, -(1:2)]), 
                dtrain,     
                nrounds = 1000, # we use early stopping
                nfold = 5,
                eval = "auc",
                objective = "binary",
                showsd = FALSE,
                stratified = TRUE,
                early_stopping_rounds = 50,
                verbose = -1)
  
  paramGrid[i, 1:2] <- as.list(cvm)[c("best_iter", "best_score")]
  print(paramGrid[i, 1:7])
  save(paramGrid, file = "paramGrid.RData") # if lgb crashes
}

# load("paramGrid.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(-paramGrid$score), ], 10)


#============================================================
# Figure out best m
#============================================================

# this validation split is necessary because lightgbm has no
# cv predictions yet
set.seed(397)
.in2 <- sample(c(FALSE, TRUE), nrow(train$X), replace = TRUE, p = c(0.3, 0.7))

dtrain2 <- lgb.Dataset(train$X[.in2, ], 
                       label = train$y[.in2])
test2 <- list(X = train$X[!.in2, ], y = train$y[!.in2])

# max m
m <- 20

# keep test predictions, no model
predTest2 <- vector(mode = "list", length = m)

for (i in seq_len(m)) {
  fit_temp <- lgb.train(paramGrid[i, -(1:2)], 
                        data = dtrain2, 
                        nrounds = floor(paramGrid[i, "iteration"] * 1.05),
                        objective = "binary",
                        verbose = -1)
  predTest2[[i]] <- predict(fit_temp, test2$X)
  print(auc(test2$y, rowMeans(do.call(cbind, predTest2[seq_len(i)]))))
}


# Use best m
m <- 5

# keep test predictions, no model
predList <- vector(mode = "list", length = m)

for (i in seq_len(m)) {
  print(i)
  gc(verbose = FALSE) # clean memory
  
  fit_temp <- lgb.train(paramGrid[i, -(1:2)], 
                        data = dtrain, 
                        nrounds = paramGrid[i, "iteration"] * 1.05,
                        objective = "binary",
                        verbose = -1)
  
  predList[[i]] <- predict(fit_temp, valid$X)
}

pred <- rowMeans(do.call(cbind, predList))
mean(round(pred) != valid$y) # 0.014


#======================================================================
# Pick glmnet because it has best performance on validation set
#======================================================================

load("data/food_glmnet.RData", verbose = TRUE)

# Evaluate "true" performance of full strategy on test data
pred <- predict(fit_glmnet, test$X, type = "response")
mean(round(pred) != test$y) # 0.014
pred[which(round(pred) != test$y)] # 36  62 114 157 183 192 289 392 704 717 750 763 871 902

#======================================================================
# Test on new data
#======================================================================

# Load the logistic regression
load("data/food_glmnet.RData", verbose = TRUE)

original_input <- preproc.images("data/check", center = mean.img)$X
deep_features <- t(adrop(predict(model2, X = original_input), 1:2))
predict(fit_glmnet, deep_features, type = "class")

# Predict label
labels[max.col(t(predict(model, X = original_input)))]


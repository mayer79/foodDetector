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
library(xgboost)

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
for (i in 2:100) {
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
# Stack of XGBoost models
#======================================================================

dtrain <- xgb.DMatrix(train$X, label = train$y)
dvalid <- xgb.DMatrix(valid$X, label = valid$y)

params <- expand.grid(max_depth = 4:6, 
                      learning_rate = c(0.6, 0.5, 0.4), 
                      subsample = c(0.6, 0.8, 1), 
                      colsample_bytree = c(0.6, 0.8, 1),
                      lambda = c(0, 0.2, 0.4, 0.6, 0.8),
                      silent = 1,
                      nthread = 8,
                      objective = "binary:logistic",
                      eval_metric = "error") #
M <- nrow(params)
eval_matrix <- matrix(NA, nrow = M, ncol = 2, dimnames = list(seq_len(M), c("iter", "metric")))

set.seed(3920)
for (i in seq_len(M)) {
  print(i)
  fit_i <- xgb.cv(params[i, ], 
                  data = dtrain, 
                  nrounds = 1000, 
                  nfold = 5, 
                  early_stopping_rounds = 3, 
                  verbose = FALSE)
  
  print(out <- fit_i$evaluation_log[fit_i$best_iteration][, c("iter", "test_error_mean")])
  eval_matrix[i, ] <- as.numeric(out)
}

# save(eval_matrix, file = "gbm_eval_matrix.RData")
#load("gbm_eval_matrix.RData", verbose = TRUE)
eval_matrix_s <- eval_matrix[order(eval_matrix[, "metric"]), ]
n_best <- 7
best <- rownames(eval_matrix_s)[seq_len(n_best)]
eval_matrix_s[best, ]
params[best, ]

# Fit top few
fit_list <- pred_list <- vector(mode = "list", length = n_best)

set.seed(303)
for (i in seq_len(n_best)) {
  # i <- 1
  print(i)
  fit_list[[i]] <- fit <- xgb.train(params[best[i], ], 
                                    data = dtrain, 
                                    nrounds = eval_matrix[best[i], "iter"],
                                    verbose = FALSE, 
                                    watchlist = list(train = dtrain))
  pred_list[[i]] <- predict(fit, dvalid)
}

pred <- rowMeans(do.call(cbind, pred_list))
mean(round(pred) != valid$y) # 0.017


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


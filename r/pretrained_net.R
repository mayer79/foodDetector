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

model <- mx.model.load("Inception/Inception_BN", iteration = 39)

# Modify the model to output the last features before going to softmax output
internals <- model$symbol$get.internals()
fea_symbol <- internals[[match("global_pool_output", internals$outputs)]]
temp <- model
temp$arg.params$fc_bias <- NULL
temp$arg.params$fc_weight <- NULL

model2 <- list(symbol = fea_symbol,
               arg.params = temp$arg.params,
               aux.params = temp$aux.params)

class(model2) <- "MXFeedForwardModel"

# Load in the mean image, which is used for preprocessing using:
mean.img <- as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])


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
pr <- princomp(train$X, cor = TRUE)

# Find number of PCs via CV
for (i in 1:200) {
  print(i)
  fit_glmnet <- cv.glmnet(x = pr$scores[, seq_len(m), drop = FALSE], 
                          y = factor(train$y), 
                          family = "binomial", 
                          lambda = c(0, 0.1), 
                          alpha = 0,
                          nfolds = 5)
  cat("cvm: ", fit_glmnet$cvm[2])
}

m <- 84 # Optimum value seems to be 84
dat <- data.frame(y = factor(train$y), pr$scores[, seq_len(m), drop = FALSE])

fit_lr <- glm(y ~ ., data = dat, family = "binomial")
pred <- predict(fit_lr, dat, type = "response")
print(mean(round(pred) != train$y)) # 0.01633333

# Predict validation data
valid$pcaScores <- as.data.frame(predict(pr, valid$X)[, seq_len(m)])
pred <- round(predict(fit_lr, data.frame(valid$pcaScores), type = "response"))
mean(pred != valid$y) # 0.022


#======================================================================
# Penalized logistic regression
#======================================================================

# Tune elastic net parameters on validation data
for (a in seq(0, 1, by = 0.1)) {
  print(a)
  fit_glmnet <- cv.glmnet(x = train$X,
                          y = factor(train$y), 
                          family = "binomial", 
                          alpha = a, 
                          nfolds = 5)
  cat("cvm: ", min(fit_glmnet$cvm))
  cat("lambda: ", fit_glmnet$lambda.min)
}

# Optimal values seem to be alpha 0.1, lambda = 0.008077715
fit_glmnet <- glmnet(x = train$X, 
                     y = factor(train$y), 
                     family = "binomial", 
                     alpha = 0.1, 
                     lambda = 0.008077715)
pred <- predict(fit_glmnet, valid$X, type = "class") 
print(mean(pred != valid$y)) # 0.014


#======================================================================
# Random forest 
#======================================================================

# OOB optimized mtry
dat <- data.frame(y = factor(train$y), train$X)
for (i in seq(1, 45, by = 1)) {
  print(i)
  fit_rf <- ranger(y ~ ., data = dat, mtry = i, num.trees = 100)
  print(fit_rf$prediction.error) # 1.5% OOB
}

# Optimium seems to be mtry = 40
fit_rf <- ranger(y ~ ., 
                 data = dat, 
                 mtry = 40, 
                 seed = 4384, 
                 importance = "impurity")
fit_rf # 0.0157
sort(importance(fit_rf)) #  X351 with highest contribution

# Predict validation data
pred <- predict(fit_rf, data = data.frame(valid$X))$predictions
mean(pred != valid$y) # 0.018

# save(fit_rf, file = "data/food_rf.RData")
load("data/food_rf.RData", verbose = TRUE)


#======================================================================
# Stack of XGBoost models
#======================================================================

dtrain <- xgb.DMatrix(train$X, label = train$y)
dvalid <- xgb.DMatrix(valid$X, label = valid$y)

params <- expand.grid(max_depth = c(1, 3, 5), 
                      learning_rate = c(0.1, 0.05, 0.01), 
                      subsample = c(0.6, 0.8, 1), 
                      colsample_bytree = c(0.6, 0.8, 1),
                      alpha = c(0, 0.2, 0.4, 0.6, 0.8),
                      silent = 1,
                      nthread = 8,
                      objective = "binary:logistic",
                      eval_metric = "error") # 0.69315 means 50% error rate
M <- nrow(params)
eval_matrix <- matrix(NA, nrow = M, ncol = 2, dimnames = list(seq_len(M), c("iter", "metric")))

# for (i in seq_len(M)) {
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
load("gbm_eval_matrix.RData", verbose = TRUE)
eval_matrix_s <- eval_matrix[order(eval_matrix[, "metric"]), ]
n_best <- 7
best <- rownames(eval_matrix_s)[seq_len(n_best)]
eval_matrix_s[best, ]
params[best, ]

# Fit top few
fit_list <- pred_list <- vector(mode = "list", length = n_best)

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
mean(round(pred) != valid$y) # 0.02


#======================================================================
# Pick glmnet because it has best performance on validation set
#======================================================================

# Evaluate "true" performance of full strategy on test data
pred <- predict(fit_glmnet, test$X, type = "class")
mean(pred != test$y) # 0.016


#======================================================================
# Test on new data
#======================================================================

dir("data/check")
check <- preproc.images("data/check", center = mean.img)
check_ <- t(adrop(predict(model2, X = check$X), 1:2))
predict(fit_glmnet, check_, type = "class")


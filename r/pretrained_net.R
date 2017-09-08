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
# library(xgboost)

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

train <- preproc.images("data/training", center = mean.img)
train$X <- t(adrop(predict(model2, X = train$X), 1:2))
gc()
valid <- preproc.images("data/validation", center = mean.img)
valid$X <- t(adrop(predict(model2, X = valid$X), 1:2))
gc()
test <- preproc.images("data/evaluation", center = mean.img)
test$X <- t(adrop(predict(model2, X = test$X), 1:2))
gc()

save(train, valid, test, file = "data/food_pretrained.RData")

# load("data/food_pretrained.RData", verbose = TRUE)

#======================================================================
# PCA and logistic regression
#======================================================================

# PCA to reduce dimension to m
m <- 100
pr <- princomp(train$X, cor = TRUE)
train$pcaScores <- as.data.frame(pr$scores[, seq_len(m), drop = FALSE])

# Classification model on first m PCAs
dat <- data.frame(y = factor(train$y), train$pcaScores)
fit_lr <- glm(y ~ ., data = dat, family = "binomial")
predict(fit_lr, dat[c(1, 2500), ], type = "response")

# Predict validation data
valid$pcaScores <- as.data.frame(predict(pr, valid$X)[, seq_len(m)])
pred <- round(predict(fit_lr, data.frame(valid$pcaScores), type = "response"))
mean(pred != valid$y) # 0.03109328

#======================================================================
# Random forest 
#======================================================================

# Classification model 
dat <- data.frame(y = factor(train$y), train$X)
fit_rf <- ranger(y ~ ., data = dat)
fit_rf # 1.5% OOB
predict(fit_rf, data = dat[c(1, 2500), ])$predictions

# Predict validation data
pred <- predict(fit_rf, data = data.frame(valid$X))$predictions
mean(pred != valid$y) # 0.01604814

# save(fit_rf, file = "data/food_rf.RData")
load("data/food_rf.RData", verbose = TRUE)


#======================================================================
# Stacks of xgboost models not shown here for brevity
#======================================================================


#======================================================================
# Pick random forest strategy over PCA + logistic regression
#======================================================================

# Evaluate "true" performance of full strategy on test data
pred <- predict(fit_rf, data = data.frame(test$X))$predictions
mean(pred != test$y) # 0.016

#======================================================================
# Test on new data
#======================================================================

check <- preproc.images("data/check", center = mean.img)
system.time(check_ <- t(adrop(predict(model2, X = check$X), 1:2)))
out <- predict(fit_rf, data = data.frame(check_))$predictions
out

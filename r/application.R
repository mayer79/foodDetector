#======================================================================
# Apply pre-trained net
#======================================================================

#======================================================================
# Required packages
#======================================================================

library(mxnet)
library(imager)
library(abind)
library(glmnet)

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
labels <- substring(read.delim("Inception/synset.txt", header = FALSE)$V1, 11)


#======================================================================
# Load the penalized logistic regression
#======================================================================

load("data/food_glmnet.RData", verbose = TRUE)


#======================================================================
# Test on new data
#======================================================================

original_input <- preproc.images("data/check", center = mean.img)$X

# Predict label
getTopM <- function(x, m = 5) {
  picks <- order(-x)[seq_len(m)]
  paste(round(x[picks], 2), labels[picks], sep = "-")
}
apply(predict(model, X = original_input), 2, getTopM)

# Predict food/non-food
deep_features <- t(adrop(predict(model2, X = original_input), 1:2))
predict(fit_glmnet, deep_features, type = "class")



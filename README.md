# Food/non-food detector

## Introduction
While many statisticians work with R on Windows, this combination is rather atypical in the field of image recognition and deep learning. 

The main reasons are as follows:

1. The most famous convolutional neural net (CNN) implementations (e.g. Caffe, Theano, Tensorflow) do not provide R bindings but rather focus on Python and/or Matlab.

2. Strong implementations of CNNs like Tensorflow are not compatible with Windows, e.g. due to the difficulties with multithreading under Windows.

To my knowledge, the `mxnet` project is the only competitive CNN implementation also running under Windows and also providing R bindings, amongst others. While 
still relatively unknown, the popularity of `mxnet` is growing, especially also in the Python community thanks to its extensive bindings. 

The aim of this article is to provide a real-world example of using `mxnet` in R Windows.

### About `mxnet`
[mxnet](https://mxnet.incubator.apache.org/) is a very efficient C implementation of different (deep) neural net architectures such as

- MLP

- CNN

- LSTM

It is being developed by the DMLC team around Tianqi Chen that is famous for the ingenious `xgboost` project.

### About CNNs
A CNN is a basic architecture of a neural net suitable for image recognition. Check out the references in [Wiki](https://de.wikipedia.org/wiki/Convolutional_Neural_Network).

## Preparation of our food detector project
### Step 1: Clone project
Open RStudio and clone this project from github. Besides subfolder "R", create two empty folders

- "data"

- "inception" (will store the pretrained neural net)

### Step 2: Installation of `mxnet` and further packages in R
Installation of mxnet is very simple and you don't need admin rights. To install the CPU version, run the R code on
https://github.com/apache/incubator-mxnet/tree/master/R-package

Further packages required are `imager`, `abind` (both for image preparation) and `ranger` (random jungle, used for classification).

If you are happy enough to have an NVIDIA graphics card, then you can even install a fully built GPU version of `mxnet`. No need to compile, no need to get CUDNN and CUDA, no need to have Visual Studio. This is amazing. Just run the corresponding lines as described in (https://mxnet.incubator.apache.org/get_started/install.html) by clicking on the right configuration.

### Step 4: The food data
Since we want to (ab)use `mxnet` to detect whether a picture shows food or not, visit this [EPFL-Site](http://mmspg.epfl.ch/food-image-datasets) and fetch the "food-5k" data set containing one zip file with three folders:

- training: 3000 pics for training

- validation: 1000 pics for model optimization

- evaluation: Another 1000 pics as test data set

Extract to folder "data". Picture names starting with "0" are non-food, those with "1" are food, e.g. like this

![ScreenShot](https://raw.github.com/mayer79/foodDetector/master/epfs_screenshot.PNG)

### Step 5: The pretrained net
In total 5000 pics are probably not enough to calculate a full fledged neural net from scratch. A smart alternative is to load a pretrained deep CNN that has been trained on the huge 2012 ImageNet data set (1.2 Mio distinct images) discriminating 1000 different objects.

One of the strong ImageNet CNNs is the Inception (V3) net or "GoogLeNet" that has nine inception modules. These are parallel convolutional elements containing differently sized filters. By this construction, a top 5 accuracy of about 3.5% is reached on the ImageNet validation set. A pretrained mxnet version of this architecture is hosted on http://data.mxnet.io/mxnet/data/Inception.zip. Get this file and extract it, so that subfolder "inception" now contains at least the following files:

1. "Inception_BN-0039.params": 40 MB of network weights...

2. "mean_224.nd": Picture intensities of the mean image used during training of the inception model 

3. "Inception_BN-symbol.json": A JSON representation of the architecture (we don't need that, but it is impressive)


## Let's start with the work
Check out the following [Tutorial](https://mxnet.incubator.apache.org/tutorials/r/classifyRealImageWithPretrainedModel.html) to get an impression how such pretrained mxnet model is used to predict the object label.

Our workflow will be the following:

1. "Hack" the inception model to be able to predict the last hidden layer (a "low" dimensional representation of the original 150'528 columns (width x height x 3 (color channels). Here, low means 1024 ;). This hack won't be necessary anymore in the future. In python, this functionality is already available, unlike in R.

2. Prepare images so they can be predicted by our inception model

3. Predict the 1024 dimensional representation of the original picture values per picture

4. Use the 1024 columns to predict food/non-food by a random forest. Random forests are especially useful here because of the (still) much too high dimension of each image. I have tested different alternatives like PCA followed by logistic regression or PCA followed by stacked grid-searched gradient boosters. They were clearly worse than the simple, untuned random forest.

5. Evaluate predictive accuracy on the test set

6. Pick some random pictures on the web to test the model

We will now open "r/pretrained_net.R" to go through that process step by step.

### Step 1: "Hack" the inception model and load the mean image
Following web instructions, this is how you remove the last (output) layer of the network to be able to predict the hidden features.

```
#======================================================================
# Required packages
#======================================================================

library(mxnet)
library(imager)
library(abind)
library(ranger)
# library(xgboost)

#======================================================================
# Load and prepare the pretrained Inception V3 GoogLeNet
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
```

The object `model` contains the original pretrained net, while `model2` stops at the last hidden layer. We will only use `model2` from now on.


### Steps 2 & 3: Prepare images and predict hidden features

Let's first load the necessary picture preparation functions.

```
source("r/functions.R")
```

For each picture folder, we first prepare all pictures (takes some minutes) and extract the 0-1 response according to the file names. Then, we predict the hidden features (takes longer than data preparation).

```
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
```

Since this step is very time consuming (ca. 1h on my laptop/PC), don't forget to store the resulting three objects on the disk!

### Steps 4: Train random forest

Now we are back in the world of classic statistics: A binary response $y$ (no food/food) and a data matrix $X$. We use the extremely fast package `ranger` to fit a binary random forest. It takes only a few seconds for training. Since the data set is not too big, we can easily save the resulting model as "fit_rf". 

```
# Classification model 
dat <- data.frame(y = factor(train$y), train$X)
fit_rf <- ranger(y ~ ., data = dat)
fit_rf # 1.5% OOB
predict(fit_rf, data = dat[c(1, 2500), ])$predictions

# Predict validation data
pred <- predict(fit_rf, data = data.frame(valid$X))$predictions
mean(pred != valid$y) # 0.01604814

save(fit_rf, file = "data/food_rf.RData")
# load("data/food_rf.RData", verbose = TRUE)
```

This gives a very low classification error on the validation set (1.6%). As usual when working with cross-sectional data, the random forest OOB estimate of the performance is very accurate (1.5%). 

### Step 5: Evaluate result on test data

On the 1000 test images, the accuracy of the validation set is retained. But, still twice as high as the stunning 0.6% of EPFL.

```
# Evaluate "true" performance of full strategy on test data
pred <- predict(fit_rf, data = data.frame(test$X))$predictions
mean(pred != test$y) # 0.016
```

### Step 6: Evaluate result

Now, you can download from the web some images and put them in subfolder "check" of folder "data" and run the following script:

```
check <- preproc.images("data/check", center = mean.img)
check_ <- t(adrop(predict(model2, X = check$X), 1:2))
predict(fit_rf, data = data.frame(check_))$predictions
```

I was grabbing the following three pictures:

![ScreenShot](https://raw.github.com/mayer79/foodDetector/master/examples_screenshot.PNG)

The result was 0-0-1, which seems correct!


## Fit your own CNN
Maybe you are disappointed: We used a pretrained neural net and did not fit our own. Shame on us! Of course this is no problem with `mxnet`. But having only 3000 images in our training set is a harsh limitations because realistic CNNs usually come with hundered thousands of parameters. In our situation, this would end up with endless overfitting issues (even using dropout after the convolution steps). One way to reduce the overfitting issue would be to create dozends of versions of each image (by flipping, slightly changing colors, picking different sections of the pictures etc.). But your laptop will be already working too hard on the original 3000 pics...

Let's open the script "r/customized_net.R".

### Data preparation

Let's load packages, functions and prepare images.

```
#======================================================================
# Required packages
#======================================================================

library(mxnet)
library(imager)
library(abind)


#======================================================================
# Some functions
#======================================================================

source("R/functions.R")


#======================================================================
# Prepare data
#======================================================================

train <- preproc.images("data/training", size = 60)
center <- apply(train$X, 1:3, mean)
train$X <- sweep(train$X, 1:3, STATS = center)
valid <- preproc.images("data/validation", size = 60)
valid$X <- sweep(valid$X, 1:3, STATS = center)
test <- preproc.images("data/evaluation", size = 60)
test$X <- sweep(test$X, 1:3, STATS = center)
save(train, valid, test, file = "data/food_customized.RData")

# load("data/food_customized.RData", verbose = TRUE)
```
Why compressing to 60 pixels per side? The reason is memory: The training data set alone would be around 6 GB of data if we would use the same picture size as GoogLeNet.

### Model specification
Next, we can specify a simple CNN with two convolutions.

```
# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=10)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=20)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc with dropout
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=50)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
dropout <- mx.symbol.Dropout(data = tanh3, p = 0.3)
# second fullc
fc2 <- mx.symbol.FullyConnected(data=dropout, num_hidden=2)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)
```

### Model fit
On CPU, running three iterations takes more than a minute. On an NVIDIA GeForce GTX 1080, it is almost 100 times faster!

```
mx.set.seed(0)
device <- mx.cpu() # device <- mx.gpu()
system.time(model <- mx.model.FeedForward.create(lenet, X = train$X, y = train$y,
                                                 ctx = device, num.round = 3, array.batch.size = 100,
                                                 learning.rate = 0.05, momentum = 0.9, wd = 0.00001,
                                                 eval.metric = mx.metric.accuracy,
                                                 batch.end.callback = mx.callback.log.train.metric(1),
                                                 epoch.end.callback = mx.callback.log.train.metric(100)))

# Evaluate on validation set
pred <- round(t(predict(model, valid$X))[, 2])
mean(pred != valid$y) #  0.22

# System time: 58.57 seconds on CPU, 0.7 seconds on GPU 
```

The accuracy is very bad, only 78% compared to our first approach using the pretrained net in combination with the random forest. Can you improve it? But don't overfit on the test set!

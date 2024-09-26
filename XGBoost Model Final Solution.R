library(xgboost) #for fitting the xgboost model
library(caret)   #for general data preparation and model fitting
library(mlr)
library(ggplot2)
library(summarytools)

Central2024P <- read.csv("Central2024P.csv", stringsAsFactors = TRUE)
summary(Central2024P)
str(Central2024P)
View(Central2024P)
attach(Central2024P)

set.seed(9876)

sum(is.na(Central2024P)) # to check whether NA exists

dim(Central2024P)

# Plot the distribution of the housing prices
ggplot(Central2024P, aes(x = Price)) + geom_histogram(aes(y = after_stat(density)), binwidth = 15000, colour = "black", fill = "white") + geom_density(alpha = .2, fill="#FF6666") +
  ggtitle("Histogram and Density Plot of House Prices") + xlab("House Price") + ylab("Frequency")

descr(Central2024P$Price)

train <- sample(1:nrow(Central2024P), 0.7*nrow(Central2024P)) # 70-30 split of train & test dataset respectively

Central2024P.train <- Central2024P[train,] # 70% train dataset
Central2024P.test <- Central2024P[-train,] # 30% test dataset

dim(Central2024P.train)
dim(Central2024P.test)

x.train <- model.matrix(Price~., data=Central2024P.train)[,-1] # Converts categorical vars to numerical dummy vars AKA "one hot encoding". https://win-vector.com/2014/06/10/r-minitip-dont-use-data-matrix-when-you-mean-model-matrix/
y.train <-Central2024P.train$Price
full.train.data <- xgb.DMatrix(data = x.train, label = y.train)

x.test <- model.matrix(Price~., data=Central2024P.test)[,-1] # Converts categorical vars to numerical dummy vars AKA "one hot encoding".
y.test <- Central2024P.test$Price
full.test.data <- xgb.DMatrix(data = x.test, label = y.test)

x <- model.matrix(Price~., data=Central2024P)[,-1]
y <- Central2024P$Price

# XGBoost
# Links: https://www.statology.org/xgboost-in-r/  &&  https://gustiyaniz.medium.com/house-price-prediction-with-xgboost-using-r-markdown-d4f891d1f327

# Define parameters
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.3,  # Learning rate.
  gamma = 5.72,
  max_depth = 8,
  min_child_weight = 1.9,
  subsample = 0.892,
  colsample_bytree = 0.777
)
# nrounds: Should be tuned using CV
# eta: Range 0-1. Typically 0.01-0.3. Lower eta leads to slower computation. It must be supported by increase in nrounds.
# gamma: Start with 0 and check CV error rate. If you see train error >>> test error, bring gamma into action. Higher the gamma, lower the difference in train and test CV. If you have no clue what value to use, use gamma=5 and see the performance. Remember that gamma brings improvement when you want to use shallow (low max_depth) trees.
# max_depth: Should be tuned using CV
# min_child_weight: Should be tuned using CV
# subsample: Control the number of features (variables) supplied to a tree. Typically 0.5-0.9

# XGBoost Hyperparameter Tuning
# Link: https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/  &&  https://xgboost.readthedocs.io/en/stable/parameter.html
# Calculate the best nround using CV = 116
xgb.cverror.out <- xgb.cv( params = params, data = full.train.data, nrounds = 500, nfold = 10, showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)
min(xgb.cverror.out$evaluation_log$test_rmse_mean)
xgb.cverror.out$best_iteration

# Random/Grid Search Procedure
traintask <- makeRegrTask(data = as.data.frame(Central2024P.train), target = "Price")
testtask <- makeRegrTask(data = as.data.frame(Central2024P.test), target = "Price")

# Do one hot encoding
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures(obj = testtask)

#create learner
lrn <- makeLearner("regr.xgboost", predict.type = "response") # Regression: “response” (= mean response) or “se” (= standard errors and mean response)
lrn$par.vals <- list(objective="reg:squarederror", eval_metric="rmse", nrounds=100L, eta=0.1)

#set parameter space
test.params <- makeParamSet(makeDiscreteParam("booster",values = c("gbtree","gblinear")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1), makeNumericParam("gamma",lower = 3L,upper = 10L)) # https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.xgboost.html

#set resampling strategy
rdesc <- makeResampleDesc("CV", iters=5L)

#random search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend to ensure faster computation
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = mse, par.set = test.params, control = ctrl, show.info = T) # measure using mse (mean of squared errors)
mytune

# define watchlist
watchlist <- list(train = full.train.data, test = full.test.data)

# fit XGBoost model and display training and testing data at each round
xgb1.fit <- xgb.train(params = params, data = full.train.data, watchlist = watchlist, nrounds = 166, print_every_n = 10, early_stopping_rounds = 10, maximize = F)

# define final model and get prediction Y
xgb.pred <- predict(xgb1.fit, full.test.data)

# MSE = 188954614634
mean((xgb.pred-y.test)^2)
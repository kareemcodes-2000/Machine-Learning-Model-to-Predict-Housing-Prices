library(glmnet)
library(leaps)
library(rpart)


Central2024P <- read.csv("Central2024P.csv", stringsAsFactors = TRUE)
summary(Central2024P)
attach(Central2024P)

set.seed(9876)

sum(is.na(Central2024P)) # to check whether NA exists

dim(Central2024P)

train <- sample(1:nrow(Central2024P), 0.7*nrow(Central2024P)) # 70-30 split of train & test dataset respectively

Central2024P.train <- Central2024P[train,] # 70% train dataset
Central2024P.test <- Central2024P[-train,] # 30% test dataset

dim(Central2024P.train)
dim(Central2024P.test)

x.train <- model.matrix(Price~., data=Central2024P.train)[,-1]
y.train <-Central2024P.train$Price

x.test <- model.matrix(Price~., data=Central2024P.test)[,-1]
y.test <- Central2024P.test$Price

x <- model.matrix(Price~., data=Central2024P)[,-1]
y <- Central2024P$Price

# Lasso Model
lasso.mod <- glmnet(x.train, y.train, alpha=1) # alpha = 1 for Lasso
lasso.cverror.out <- cv.glmnet(x.train, y.train, alpha=1, K=10) # get 10 MSE
plot(lasso.cverror.out)
lasso.best.lambda <- lasso.cverror.out$lambda.min # get the best lambda for the smallest cv error
lasso.best.lambda
lasso.pred <- predict(lasso.mod, newx=x.test, s=lasso.best.lambda)

# MSE = 439790515474
mean((lasso.pred-y.test)^2)

full.lasso.mod <- glmnet(x,y, alpha=1)
full.lasso.coef <- predict(full.lasso.mod, type="coefficients", s=lasso.best.lambda)[1:19,]
full.lasso.coef
full.lasso.coef[full.lasso.coef !=0]

# Best model: Price = 310670.447 + 2160.763 Area - 36084.729 Age - 354972.405 Tenure(Leasehold) + 98948.096 Region(Bukit Merah) - 89795.894 Region(Geylang)
# + 234545.211 Region(Marine Parade) + 1546260.270 Region(Newton) + 100374.194 Region(Novena) + 725046.247 RegionOthers - 19333.868 Region(Queenstown)
# + 917492.183 Region(River Valley) - 9817.901 Region(Rochor) + 44304.357 Region(Southern Islands) + 791059.168 Region(Tanglin) - 81984.488 Region(Toa Payoh)

# Ridge Model
ridge.mod <- glmnet(x.train, y.train, alpha=0) # alpha = 0 for Ridge Regression
ridge.cverror.out <- cv.glmnet(x.train, y.train, alpha=0, K=10)
plot(ridge.cverror.out)
ridge.best.lambda <- ridge.cverror.out$lambda.min # get the best lambda for the smallest cv error
ridge.best.lambda
ridge.pred <- predict(ridge.mod, newx=x.test, s=ridge.best.lambda)

# MSE = 478012111963
mean((ridge.pred-y.test)^2)

full.ridge.mod <- glmnet(x,y, alpha=0)
full.ridge.coef <- predict(full.ridge.mod, type="coefficients", s=ridge.best.lambda)[1:19,]
full.ridge.coef
full.ridge.coef[full.ridge.coef !=0]

# Best model: Price = 526353.245 + 1928.296 Area - 27553.052 Age - 371720.345 Tenure(Leasehold) + 53057.902 Purchaser(Private) + 54569.672 Region(Bukit Merah) - 74666.834 Region(Bukit Timah)
# - 187380.251 Region(Geylang) - 83008.328 Region(Kallang) + 167456.509 Region(Marine Parade) + 1492824.598 Region(Newton0 + 28010.105 Region(Novena) + 615893.105 Region(Others)
# - 137561.229 Region(Queenstown) + 859062.870 Region(River Valley) - 122147.414 Region(Rochor) + 227807.344 Region(Southern Islands) + 696093.694 Region(Tanglin) - 152426.469 Region(Toa Payoh)

# Decision Tree Model
decision.tree <- rpart(Price~., method = "anova", data = Central2024P[train,])
decision.tree
decision.tree.predict <- predict(decision.tree, Central2024P[-train,], method = "anova")

# MSE = 664782136998
mean((decision.tree.predict-Central2024P[-train,]$Price)^2)

# Linear Regression Models
#glm1.mod <- glm(Price~Area + Age + Tenure + Region, data=Central2024P)
#glm1.cverror.out <- cv.glm(Central2024P, glm1.mod)
#glm1.cverror.out

## Testing
#get_mse <- function(lm) 
#  mean(lm$residuals^2)

## Linear Regression Model
#lm.fit1 <- glm(Price~Area + I(Area^2) + Age + Tenure + Region, data=Central2024P)
#summary(lm.fit1)
#get_mse(lm.fit1)
#BIC(lm.fit1)
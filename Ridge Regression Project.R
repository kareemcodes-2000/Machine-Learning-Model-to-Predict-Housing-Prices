library(leaps)
library(glmnet)

central <- read.csv("Central2024P.csv", stringsAsFactors = TRUE)

set.seed(9876)

sum(is.na(central))
dim(central)

train <- sample(1:nrow(central), 0.7*nrow(central))

central.train <- central[train,]
central.test <- central[-train,]


train.x <- model.matrix(Price~., data = central.train)[,-1]
train.y <- central.train$Price

test.x <- model.matrix(Price~., data = central.test)[,-1]
test.y <- central.test$Price

x <- model.matrix(Price~., data = central)[,-1]
y <- central$Price

ridge.mod <- glmnet(train.x,train.y, alpha = 0)
cv.out <- cv.glmnet(train.x, train.y, alpha = 0, K = 10)
lambda.rr <- cv.out$lambda.min
lambda.rr

ridge.pred <- predict(ridge.mod, newx = test.x, s = lambda.rr)
plot(ridge.pred)

mean((ridge.pred - test.y)^2)

#MSE: 478012111963

out.rr <- glmnet(x,y, alpha = 0)
rr.coef <- predict(out.rr, type = "coefficients", s = lambda.rr)[1:19,]
rr.coef
rr.coef[rr.coef !=0]

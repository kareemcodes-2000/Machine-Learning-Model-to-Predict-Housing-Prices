library(car)
library(ISLR)
library(fitdistrplus)
library(lmtest)
library(caret)
library(MASS)
setwd("~/Desktop/DataSets")
central<-read.csv("Central2024P.csv",stringsAsFactors = TRUE)
attach(central)
str(central)

l1 <- lm(Price~.,data=central)
summary(l1)
#Purchaser is insignificant, drop variable. 
l2 <- lm(Price~Area+Age+Tenure+Region, data=central)
anova(l2,l1) ##0.9793>0.05 , insignificant, therefore definitely can drop the variable
summary(l2)

#Check for Outliers
head(central)
boxplot(central$Area)
boxplot(central$Price)

#Check for multicollinearity
vif(l1) #VIFs < 5, no multicollinearity

# Check full model in interaction terms
l3 <- lm(formula = Price~(Area+Age+Tenure+Region)^2, data=central)
summary(l3)

pairs(formula = Price~(Area+Age+Tenure+Region)^2, data=central, panel=panel.smooth)
#Variables are more or less linear, no need for higher order. 

l4 <- lm(formula = Price~Area*(Region+Age+Tenure), data=central)
summary(l4)


c(BIC(l1),BIC(l2),BIC(l3),BIC(l4))

#Testing for homoscedasticity
bptest(l4) #2.2e-16<0.05, reject null hypothesis. No homoscedasticity.

# Removing Outliers
Q <- quantile(central$Price, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(central$Price)
up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range

central1<- subset(central, central$Price > (Q[1] - 1.5*iqr) & central$Price < (Q[2]+1.5*iqr)) 
summary(central)

Q <- quantile(central1$Area, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(central1$Area)
up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range
central2<- subset(central1, central1$Area > (Q[1] - 1.5*iqr) & central1$Area < (Q[2]+1.5*iqr))
summary(central2)

# Create the training and testing datasets
set.seed(9876)
train_indices <- sample(1:nrow(central2), size = round(0.7 * nrow(central2)))
train_data <- central2[train_indices, ]
test_data <- central2[-train_indices, ]

dim(train_data)
dim(test_data)

# Running model
model <- lm(formula = Price~Area*(Region+Age+Tenure), data = train_data)
summary(model)
BIC(model) 
# Test for residual
res <- resid(model)
plot(fitted(model), res)
abline(0,0)

# Test for MSE 
get_mse <- function(lm) 
  mean((lm$residuals)^2)

get_mse(model) 
#70461539507



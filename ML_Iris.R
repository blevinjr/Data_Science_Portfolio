##############################
##                          ##
## Machine Learning Project ##
##                          ##
##############################

#clear all
rm (list = ls())

#Add directories for statistical modeling/ts/and machine learning. More packages than needed have been uploaded just in case something else is required of the data that might need one of these libraries.
library(haven)
library(modelr)
library(dplyr)
library(zoo)
library(lmtest)
library(sandwich)
library(data.table)
library(car)
library(prais)
library(olsrr)
library(aTSA)
library(rugarch)
library(DataCombine)
library(AER)
library(smooth)
library(forecast)
library(vars)
library(egcm)
library(caret)
library(ggplot2)
library(readxlsx)
library(ellipse)
library(e1071)
library(randomForest)
library(shiny)
library(tidyverse)

setwd("Users/justinblevins/download") #The Place I'll be pulling data from


#loading data
data(iris)
#renaming data set
dataset <- iris

#creating a training set containing 80% of the rows in the original dataset.
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)

#selecting 20% of the data to be used for validation
validation <- dataset[-validation_index,]

#using 80% of data for training and testing the models
dataset <- dataset[validation_index,]


#dimensions of dataset
dim(dataset)

#listing types for each attribute
sapply(dataset, class)

#checking the first 5 rows of the data
head(dataset)

#listing the levels for the class
levels(dataset$Species)


#summarizing class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

#statistical summary of dataset
summary(dataset)

#splitting input and output
x <- dataset[,1:4]
y <- dataset[,5]

#creting a boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}


#creating a barplot for class breakdown
plot(y)

#creating a scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

#creating a box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

#creating density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#To estimate accuracy 10-fold crossvalidation is used which splits the dataset into 10 parts, it trains with 9 and tests on 1 then releases all combinations of the training-testing splits. This process will get repeated 3 times for each algorithm with different splits of the data into 10 groups, in hopes of acheiving a more accurate estimate.

#Running the algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#The following 5 different algorithms will be used and then the best one will be determined: simple linear (LDA), nonlinear (CART, kNN) and complex nonlinear methods (SVM, RF). In order to ensure each algorithm has the same data splits which will allow for direct comparability, the random number seed will be reset before each execution of each algorithm.

#linear algorithm
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

#nonlinear algorithm: CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

#nonlinear algorithm: kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

#advanced algorithm: SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)

#advanced algorithm: Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)


#Creating a list of the models so they can be summarized to determine which is the most accurate.
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))

#compare accuracy of models with visualization
dotplot(results)

#summarize 'results' to get the accuracy of the models.
summary(results)

#Comparing the spread and the mean accuracy of each model. The ten fold cross-validation gives us a population of accuracy measures for each algorithm since it was evaluated 10 times.

#The most accurate model turns out to be the LDA model.

#View a summary of just the LDA model. This will display a summary of what was used to train the model as well as the mean and standard deviation of the accuracy.
print(fit.lda)

#This model gives us 97.5% accuracy +/- 4%


#To ensure we don't have a result that is derived from overfitting or a data leak which might lead to misleadingly high accuracy we will run the LDA on the validation set and summarize the results in a confusion matrix.
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

#The accuracy shows 100% and although the validation set was small at 20%, the 100% accuracy obtained from the 'predictions/confusionMatrix' commands is within the 4% margin which means it is likely to be a reliably accurate model.


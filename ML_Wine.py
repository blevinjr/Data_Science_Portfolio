###################################
##                               ##
##  Data Cleaning: Data Science  ##
##                               ##
###################################

# Importing necessary packages to complete the project. Further package breakdown noted below
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing sampling helper
from sklearn.model_selection import train_test_split

#importing processing modules
from sklearn import preprocessing

#import random forests models
from sklearn.ensemble import RandomForestRegressor

#Tools for performing cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#metrics to evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score

#Module for saving scicit-learn models
from sklearn.externals import joblib

#Loading data set
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url)

#Checking out the first 5 rows of data
print data.head()

#Cleaning up the data so it is easier to view
data = pd.read_csv(dataset_url, sep=';')
print data.head()

#Taking a look at the data
print data.shape

#Obtaining summary statistics
print data.describe()

#All data is numeric, however, there are different scales present. I will standardize the data later
#Further exploratory measures should be taken to gain a more full understanding of the data

#Seperating the target variable from the training features
y = data.quality
X = data.drop('quality', axis=1)

#Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)

#Fitting a transformer API
scaler = preprocessing.StandardScaler().fit(X_train)

#Applying transformer to training set
X_train_scaled = scaler.transform(X_train)

 #Checking to see if transformer worked
print X_train_scaled.mean(axis=0)
print X_train_scaled.std(axis=0)

#With code above it takes the scaler object and uses it to transform the 
#training set. Later it will be used to transform the test set using the exact
#same means and standard deviations used to transform the training set

#Applying the transformer to the test set
X_test_scaled = scaler.transform(X_test)

#Checking the results
print X_test_scaled.mean(axis=0)
print X_test_scaled.std(axis=0)

#Preprocessing and model(standarizing data and fitting a model using random forrest)
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))

#Listing parameters
print pipeline.get_params()

#Declaring the hyperparameters that will be tuned through cross-validation
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

#Using Sklearn for cross-validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
#Fit and tune model
clf.fit(X_train, y_train)

#Checking which parameters are best
print clf.best_params_

#Confirming GridSearchCV automatically refit model with the best set of 
#hyperparameters(confirming model will be retrained)
print clf.refit

#Now the clf object will be used as the model when applying it to other data sets

#Predicting a new set of data
y_pred = clf.predict(X_test)

#Using metrics imported earlier to evaluate the model
print r2_score(y_test, y_pred)
print mean_squared_error(y_test, y_pred)

#Saving this model for later use
joblib.dump(clf, 'random_forest_regressor_model.pkl')



########################################################
# Code for Loading and using this model in the future ##
########################################################
# clf2 = joblib.load('rf_regressor.pkl')
# Predict data set using loaded model
# clf2.predict(X_test)


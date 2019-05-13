# -*- coding: utf-8 -*-
"""
@author: Sreenivas.J
"""
#DecissionTree and Predict methods are very important in this example. This is the real starting/building of ML
#Here we will be playing with more columns. However DecisionTreeClassifier algorithm works only on numeric/continuous data/columns
#Henceforth we need to convert  catogerical columns to dummy columns
#This technique is called one-hot encoding

import pandas as pd
from sklearn import tree
#import io
#import pydot #if we need to use any external .exe files.... Here we are using dot.exe
import os
#from sklearn import model_selection
os.chdir("D:/Data Science/Data/")

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

titanic_train = pd.read_csv("D:/Data Science/Data/titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#Convert categoric to One hot encoding using get_dummies
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.describe

#now the drop non numerical columns where we will not be applying logic. Something like we will not apply logic on names, passengerID ticket id etc...
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'],1) 
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier()
#Build the decision tree model
param_grid = {'max_depth':[8, 10, 15], 'min_samples_split':[2, 4, 6], 'criterion':['gini', 'entropy']}

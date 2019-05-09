# -*- coding: utf-8 -*-
"""
@author: Sreenivas.J
"""

import pandas as pd
from sklearn import tree
import io
import pydot #if we need to use any external .exe files....
import os
os.chdir("D:/Data Science/Data/")
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

titanic_train = pd.read_csv("D:/Data Science/Data/titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()
titanic_train.describe()
#Transformation of non numneric cloumns to 1-Hot Encoded columns
#There is an exception with the Pclass. Though it's co-incidentally a number column but it's a Categoric column(Even common-sence wise).

#Transform categoric to One hot encoding using get_dummies
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.head(10)
titanic_train1.info()
titanic_train1.describe()

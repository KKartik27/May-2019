# -*- coding: utf-8 -*-
"""

@author: Sreenivas.J
"""

import pandas as pd
import os
from sklearn import tree, ensemble
from sklearn import preprocessing
from sklearn import model_selection
#from sklearn import feature_selection
from sklearn_pandas import CategoricalImputer

from mlxtend import classifier as mlxClassifier #For stacking

os.chdir("D:/Data Science/Data")

titanic_train = pd.read_csv("titanic_train.csv")
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv("titanic_test.csv")
titanic_test.shape

titanic_all = pd.concat([titanic_train, titanic_test])
titanic_all.shape
titanic_all.info()

#impute missing values for continuous features
imputable_cont_features = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_all[imputable_cont_features])
titanic_all[imputable_cont_features] = cont_imputer.transform(titanic_all[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_all['Embarked'])
titanic_all['Embarked'] = cat_imputer.transform(titanic_all['Embarked'])

titanic_all['FamilySize'] = titanic_all['SibSp'] +  titanic_all['Parch'] + 1

def convert_family_size(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
titanic_all['FamilyCategory'] = titanic_all['FamilySize'].map(convert_family_size)

def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_all['Title'] = titanic_all['Name'].map(extract_title)

tmp_df = titanic_all[0:titanic_train.shape[0]]

titanic_all.drop(['PassengerId', 'Name', 'Cabin','Ticket','Survived'], axis=1, inplace=True)

features = ['Sex', 'Embarked', 'Pclass', 'Title', 'FamilyCategory']
titanic_all = pd.get_dummies(titanic_all, columns=features)

X_train = titanic_all[0:titanic_train.shape[0]]
y_train = titanic_train['Survived']

#build stacked model using selected features
rf1 = ensemble.RandomForestClassifier(random_state=100)
ada2 = ensemble.AdaBoostClassifier(random_state=100)

dtSuper = tree.DecisionTreeClassifier(random_state=100)

stack_estimator = mlxClassifier.StackingClassifier(classifiers=[rf1, ada2], meta_classifier=dtSuper) #, store_train_meta_features=True)
stack_grid = {'randomforestclassifier__n_estimators': [5, 10],
            'adaboostclassifier__n_estimators': [10, 50],
            'meta-decisiontreeclassifier__min_samples_split': [2, 3]}

grid_stack_estimator = model_selection.GridSearchCV(stack_estimator, stack_grid, cv=10)
grid_stack_estimator.fit(X_train, y_train)
#grid_stack_estimator.fit(X_train1, y_train)

final_model = grid_stack_estimator.best_estimator_
print(final_model.clfs_) #Classifiers
print(final_model.meta_clf_) #Meta Classifiers

     
X_test = titanic_all[titanic_train.shape[0]:]
titanic_test['Survived'] = grid_stack_estimator.predict(X_test)

titanic_test.to_csv('submission_Stacking.csv', columns=['PassengerId','Survived'],index=False)

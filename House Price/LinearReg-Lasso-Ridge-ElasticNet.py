# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:53:37 2019

@author: Sreenivas.J
"""

import os
import pandas as pd
from sklearn import preprocessing, metrics
import numpy as np
from sklearn import model_selection, ensemble
import seaborn as sns
from sklearn import feature_selection
import math
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn import linear_model

def get_continuous_features(df):
    return df.select_dtypes(include=['number']).columns

def get_categorical_features(df):
    return df.select_dtypes(exclude=['number']).columns

def cast_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')

def transform_cat_cont(df, features):
    return pd.get_dummies(df, columns = features)

def impute_continuous_features(df, features):
    cont_imputer = preprocessing.Imputer()
    cont_imputer.fit(df[features])
    #print(cont_imputer.statistics_)
    df[features] = cont_imputer.transform(df[features])
    
#From SkLearn_Pandas
def impute_categorical_features(df, features):    
    feature_defs = []
    for col_name in features:
        feature_defs.append((col_name, CategoricalImputer()))
    mapper = DataFrameMapper(feature_defs, input_df=True, df_out=True)
    df[features] = mapper.fit_transform(df[features])

def get_features_to_drop_on_missingdata(df, threshold):
    tmp = df.isnull().sum()
    return list(tmp[tmp/float(df.shape[0]) > threshold].index)

def drop_features(df, features):
    return df.drop(features, axis=1, inplace=False)

def select_features_from_model(selector, X_train, y_train):
    selector.fit(X_train, y_train)
    plot_feature_importances(selector, X_train, y_train)
    #Threshold : string, float, optional default None(    )
    select = feature_selection.SelectFromModel(selector, prefit=True)
    return select.transform(X_train)

def plot_feature_importances(estimator, X_train, y_train):
    indices = np.argsort(estimator.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],x = estimator.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("RF feature importances")   

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )
 
def fit_model(estimator, grid, X_train, y_train):
   grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse), cv=10, n_jobs=1)
   grid_estimator.fit(X_train, y_train)
   print(grid_estimator.cv_results_)
   print(grid_estimator.best_params_)
   final_model = grid_estimator.best_estimator_
#==============================================================================
#    print(final_model.coef_)
#    print(final_model.intercept_)
#==============================================================================
   print(grid_estimator.best_score_)
   print(grid_estimator.score(X_train, y_train))
   return final_model

#changes working directory
path = 'D:/Data Science/Data/'
house_train = pd.read_csv(os.path.join(path,"house_train.csv"))
house_train.shape
house_train.info()

#type cast features
features_to_cast = ['MSSubClass']
cast_cont_to_cat(house_train, features_to_cast)

#retrieve continuous & categorical features
get_continuous_features(house_train)
get_categorical_features(house_train)

#drop non-useful features
features_to_drop = ['Id', 'SalePrice']
features_to_drop.extend(get_features_to_drop_on_missingdata(house_train, 0.35))
house_train1 = drop_features(house_train, features_to_drop)
house_train1.info()
house_train1.shape
impute_categorical_features(house_train1, get_categorical_features(house_train1))
impute_continuous_features(house_train1, get_continuous_features(house_train1))

house_train1 = transform_cat_cont(house_train1, get_categorical_features(house_train1))

X_train = house_train1
y_train = house_train['SalePrice']

#Model 1
selector = ensemble.RandomForestClassifier(random_state=100)
X_train1 = select_features_from_model(selector, X_train, y_train)

#Model 2 which is OBL
linr_estimator = linear_model.LinearRegression()
linr_estimator.fit(X_train1, y_train)
coef = linr_estimator.coef_
intercept = linr_estimator.intercept_
#crossVal = model_selection.cross_val_score(linr_estimator, X_train1, y_train, cv=10, scoring=metrics.make_scorer(rmse))
crossValidate = model_selection.cross_validate(linr_estimator, X_train1, y_train, cv=10, scoring=metrics.make_scorer(rmse))
crossValidate.get('fit_time').mean()
crossValidate.get('score_time').mean()
crossValidate.get('test_score').mean()
crossValidate.get('train_score').mean()

#Calculate the Sum of the Squared(Error) values of the weights, called L2.
#It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
#It reduces the model complexity by coefficient shrinkage.
ridge_estimator = linear_model.Ridge()
ridge_grid = {'alpha':[1.0], 'max_iter':[50]}
ridgeModel = fit_model(ridge_estimator, ridge_grid, X_train1, y_train)
coef = ridgeModel.coef_
intercept = ridgeModel.intercept_

#LASSO: Least Absolute Shrinkage Selector Operator
#Calculate the sum of the Absolute values of the weights, called L1.
#Nullifies low important features with 0 co-efficient
#It is generally used when we have more number of features, because it automatically does feature selection.
lasso_estimator = linear_model.Lasso()
lasso_grid = {'alpha':[0.5, 0.9]}
lassoModel = fit_model(lasso_estimator, lasso_grid, X_train1, y_train)
coef = lassoModel.coef_
intercept = lassoModel.intercept_

#ElasticNet is a comibination of Ridge & LASSO
enet_estimator = linear_model.ElasticNet()
enet_grid = {'alpha':[0.1], 'l1_ratio':[0.3]}
fit_model(enet_estimator, enet_grid, X_train1, y_train)
import os
import pandas as pd
from sklearn.externals import joblib

#changes working directory
os.chdir("D:/Data Science/Data")

#predict the outcome using decision tree
titanic_test = pd.read_csv("titanic_test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)

#Use load method to load Pickle file
dtree = joblib.load("TitanicVer1.pkl")
titanic_test['Survived'] = dtree.predict(X_test)
titanic_test.to_csv("submissionUsingJobLib.csv", columns=['PassengerId','Survived'], index=False)

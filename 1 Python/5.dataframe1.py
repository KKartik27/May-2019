#Data Frames1
#Python is case sensitive
import pandas as pd
print(pd.__version__)
titanic_train = pd.read_csv("D:/Data Science/Data/titanic_train.csv")
print(type(titanic_train))

#explore the dataframe
titanic_train.shape #No of rows and Column
titanic_train.info() #Data Type and nullable/non-nullable
titanic_train.describe() #Gives statistical information

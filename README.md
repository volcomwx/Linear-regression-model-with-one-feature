# Linear-regression-model-with-one-feature

# 导入 import

import os

import pandas as pd 

import numpy as np

import plotly.express as px

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

# 数据探索 explore data

train = pd.read_csv('E:/Resume/机器学习数据模型/线性回归/一元线性回归方程/train.csv')

test = pd.read_csv('E:/Resume/机器学习数据模型/线性回归/一元线性回归方程/test.csv')

train.head()

|   |   x |          y |
|---|-----|------------|
| 0 | 24  | 21.549452  |
| 1 | 50  | 47.464463  |
| 2 | 15  | 17.218656  |
| 3 | 38  | 36.586398  |
| 4 | 87  | 87.288984  |

test.head()

train.shape


(700, 2)

test.shape

(300, 2)

train.isnull().sum()

x    0
y    1
dtype: int64


test.isnull().sum()

x    0
y    0
dtype: int64

train = train.dropna()

Set training data and targets

X_train = train['x']

y_train = train['y']

# Set testing data and targets

X_test = test['x']

y_test = test['y']

# reshape the data to fit in the model 将dataframe变成一个数字数值的向量

X_train = X_train.values.reshape(-1, 1)

X_test = X_test.values.reshape(-1,1)

# rescale the data to converge faster 用z-score方法把x的值进行特征缩放

# Create a StandardScaler object

scaler = StandardScaler()

# Fit the scaler to the training data and compute mean and standard deviation (z-score method)

scaler.fit(X_train)

# Transform (standardize) the training data based on the computed mean and standard deviation

X_train = scaler.transform(X_train)

# Transform (standardize) the test data using the same mean and standard deviation

X_test = scaler.transform(X_test)

print("Transformation sucessful, now both x_train and x_test were rescaled.")

Transformation sucessful, now both x_train and x_test were rescaled.

X_train.min(),X_train.max()

(-1.72857469859145, 1.7275858114641094)


# Visualize The Data 做图

from IPython.display import display

import plotly.express as px

fig = px.scatter(x=train['x'], y=train['y'], template='gridon')

# Display the figure

fig.show()

# Model 建立模型

model = LinearRegression() #Create linear regression instance

model.fit(X_train, y_train) #fit the linear regression model to the training data and labels

#evaluate the results on the test set

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions) #Get the mean squared error as the evaluation metric

print(f'the mean squared error is: {mse}')

the mean squared error is: 9.43292219203933

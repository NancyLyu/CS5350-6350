import pandas as pd
import numpy as np

train = pd.read_csv("/Users/weiranlyu/Desktop/2021 Fall/CS6350/Neural Networks/train.csv")
test = pd.read_csv("/Users/weiranlyu/Desktop/2021 Fall/CS6350/Neural Networks/test.csv")

train.loc[-1] = train.columns.values
train.sort_index(inplace=True)
train.reset_index(drop=True, inplace=True)
y_train = train.iloc[:,-1].copy().values
y_train[0] = int(y_train[0])
y_train = y_train*2 - 1
X_train = train.values
X_train[0] = [float(i) for i in X_train[0]]
X_train[:, 4] = 1

test.loc[-1] = test.columns.values
test.sort_index(inplace=True)
test.reset_index(drop=True, inplace=True)
y_test = test.iloc[:,-1].copy().values
y_test[0] = int(y_test[0])
y_test = y_test*2 - 1
X_test = test.values
X_test[0] = [float(i) for i in X_test[0]]
X_test[:, 4] = 1
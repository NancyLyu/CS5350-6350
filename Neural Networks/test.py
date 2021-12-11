import pandas as pd
import numpy as np
import SGD_NeuralNetwork as NN

train = pd.read_csv("/Users/weiranlyu/Desktop/2021 Fall/CS6350/Neural Networks/bank-note/train.csv")
test = pd.read_csv("/Users/weiranlyu/Desktop/2021 Fall/CS6350/Neural Networks/bank-note/test.csv")

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

width = [5, 10, 25, 50 ,100]
for w in width:
    s = [X_train.shape[1], w, w, 1]
    model= NN.NeuralNetwork(s)

    model.train(X_train.reshape([-1, X_train.shape[1]]), y_train.reshape([-1,1]))
    pred_train= model.fit(X_train)
    train_error = model.predict(y_train, pred_train)

    pred_test = model.fit(X_test)
    test_error = model.predict(y_test, pred_test)

    print('train_error: ', train_error, ' test_error: ', test_error)

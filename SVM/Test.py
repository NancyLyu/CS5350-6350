import pandas as pd
import numpy as np
import SVM

#process data
train = pd.read_csv("/Users/weiranlyu/Desktop/CS6350/SVM/bank-note/train.csv")
test = pd.read_csv("/Users/weiranlyu/Desktop/CS6350/SVM/bank-note/test.csv")

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

svm = SVM.SVM()
C = np.array([100/873, 500/873, 700/873])
Gamma = np.array([0.1, 0.5, 1.5, 100])
#2 Stochastic sub-gradient SVM
for c in C:
    svm.set_C(c)
    print("C is ", c, ": ")
    weights = svm.GSD(X_train, y_train)
    weights = np.reshape(weights, (5,1))

    prediction = svm.predict(X_train, weights)
    train_err = svm.predictError(prediction, y_train)

    prediction = svm.predict(X_test, weights)
    test_err = svm.predictError(prediction, y_test)
    print("SGD SVM: Training error is ", train_err, " and testing error is ", test_err)
    
    #3(a) Dual SVM
    weights = svm.dual(X_train[:,[x for x in range(4)]], y_train)
    weights = np.reshape(weights, (5,1))
    train_pred = svm.predict(X_train, weights)
    train_error = svm.predictError(train_pred, y_train)
    test_pred = svm.predict(X_test, weights)
    test_error = svm.predictError(test_pred, y_test)
    print("Dual SVM: Training error is ", train_error, " and testing error is ", test_error)

    #3(b) Gaussian kernel SVM
    count  = 0
    for gamma in Gamma:
        svm.set_gamma(gamma)
        print("gamma is ", gamma)
        x_tr = X_train[:,[x for x in range(4)]]
        alpha = svm.Gaussian(x_tr, y_train)
        index = np.where(alpha > 0)[0]
        print("Gaussian kernel SVM: The number of support vectors is", len(index))

        y_pred = svm.predictGaussian(alpha, x_tr, y_train, x_tr)
        train_error = svm.predictError(y_pred, y_train)
        
        x_te = X_test[:,[x for x in range(4)]]
        y_pred = svm.predictGaussian(alpha, x_tr, y_train, x_te)
        test_error = svm.predictError(y_pred, y_test)
        print("Training error is ", train_error, "and testing error is ", test_error)

        if count > 0:
            overlap = len(np.intersect1d(index, pre_index))
            print("The number of overlapped support vectors is ", overlap)
        count += 1
        pre_index = index


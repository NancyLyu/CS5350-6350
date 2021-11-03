import pandas as pd

#The standard Perceptron algorithm

def weightVectors(train, r, T):
    #initialize w_0
    weights = [0, 0, 0, 0]
    for t in range(T):
        for i in train:
            y_pred = predict(i, weights)
            y_true = i[-1]
            err = y_true - y_pred
            if y_true != y_pred:
                for j in range(len(i) - 1):
                    weights[j] += r*err*i[j]
    return weights

#for each training example (x_i, y_i), predict y'
def predict(row, weights):
    sign = 0
    for i in range(len(row) - 1):
        sign += row[i]*weights[i]
    return 1 if sign >= 0 else 0

def perceptron(train, test, r, T):
    weights = weightVectors(train, r, T)
    predictions = []
    for i in test:
        predictions.append(predict(i, weights))
    return predictions

def predictionError(y_true, y_pred):
    error = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            error += 1
    return error / float(len(y_true)) * 100

#clean up data
train = pd.read_csv("/Users/weiranlyu/Desktop/CS6350/Perceptron/bank-note/train.csv")
test = pd.read_csv("/Users/weiranlyu/Desktop/CS6350/Perceptron/bank-note/test.csv")

train.loc[-1] = train.columns.values
train.sort_index(inplace=True)
train.reset_index(drop=True, inplace=True)
train.columns = ["variance", "skewness", "curtosis", "entropy", "label"]

test.loc[-1] = test.columns.values
test.sort_index(inplace=True)
test.reset_index(drop=True, inplace=True)
test.columns = ["variance", "skewness", "curtosis", "entropy", "label"]

test_label = test["label"]
test_label[0] = 0
test_label = test_label.tolist()

train = train.values.tolist()
test = test.values.tolist()

for i in range(5):
    train[0][i] = float(train[0][i])
for i in range(5):
    test[0][i] = float(test[0][i])
train[0][4] = int(train[0][4])
test[0][4] = int(test[0][4])

#get the learned weight vector and average prediction error on test dataset
weights = weightVectors(train, 0.1, 10)
y_pred = perceptron(train, test, 0.1, 10)
error  = predictionError(test_label, y_pred)
print('Learned weight vector: %s' % weights)
print('Prediction error: %.3f%%' % error)

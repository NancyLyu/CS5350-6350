import pandas as pd

#The voted Perceptron algorithm

def weightVectors(train, r, T):
    #initialize w_0
    weights = [0, 0, 0, 0]
    distinctWeights = []
    c = [0]
    m = 0
    for t in range(T):
        for i in train:
            y_pred = predict(i, weights)
            y_true = i[-1]
            err = y_true - y_pred
            if y_true != y_pred:
                for j in range(len(i) - 1):
                    weights[j] += r*err*i[j]
                copy = weights.copy()
                distinctWeights.append(copy)
                m += 1
                c.append(1)
            else:
                c[m] += 1
    return distinctWeights, c

#for each training example (x_i, y_i), predict y'
def predict(row, weights):
    sign = 0
    for i in range(len(row) - 1):
        sign += row[i]*weights[i]
    return 1 if sign >= 0 else 0

def votePrediction(row, distinctWeights, c):
    sign = 0
    sign_inner = 0
    counter = 0
    for i in distinctWeights:
        for j in range(len(row) - 1):
            sign_inner += row[j]*i[j]
        if sign_inner >= 0:
            sign_inner = 1
        else:
            sign_inner = -1
        sign += sign_inner*c[counter]
        counter += 1
    return 1 if sign >= 0 else 0
        

def votePerceptron(distinctWeights, counts, test):
    predictions = []
    for i in test:
        predictions.append(votePrediction(i, distinctWeights, counts))
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
weights, count = weightVectors(train, 0.1, 10)
count = count[1:]
y_pred = votePerceptron(weights, count, test)
error  = predictionError(test_label, y_pred)
print('Distinct weight vectors: %s' % weights)
print("Number of correctly predicted training examples of each distinct weight vectors: %s" % count)
print('Prediction error: %.3f%%' % error)
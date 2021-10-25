import DecisionTree as DT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def AdaBoost(X_train, Y_train, X_test, Y_test, M, clf_tree):
    # Initialize weights
    numOfTrain, numOfTest = len(X_train), len(X_test)
    w = np.ones(numOfTrain) / numOfTrain
    predTrain, predTest = [np.zeros(numOfTrain), np.zeros(numOfTest)]
    for i in range(M):
        predTrain_i = DT.predicted(X_train, clf_tree)
        predTest_i = DT.predicted(X_test, clf_tree)
        miss = [int(x) for x in (predTrain_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)
        # Alpha
        alpha = 0.5 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha for x in miss2]))
        # Add to prediction
        predTrain_i = [1 if x == 1 else -1 for x in predTrain_i]
        predTest_i = [1 if x == 1 else -1 for x in predTest_i]
        predTrain = predTrain + np.multiply(alpha, predTrain_i)
        predTest = predTest + np.multiply(alpha, predTest_i)
    
    predTrain, predTest = np.sign(predTrain), np.sign(predTest)
    return sum(predTrain != Y_train) / float(len(Y_train)), \
           sum(predTest != Y_test) / float(len(Y_test))

def clf(X_train, Y_train, X_test, Y_test, clf_tree):
    resultXTrain = DT.predicted(X_train, clf_tree)
    resultXTest = DT.predicted(X_test, clf_tree)
    return sum(resultXTrain != Y_train) / float(len(Y_train)), \
           sum(resultXTest != Y_test) / float(len(Y_test))


def plot_error_rate(errorTrain, errorTest):
    df_error = pd.DataFrame([errorTrain, errorTest]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Error rate vs number of iterations', fontsize = 16)
    plt.axhline(y=errorTest[0], linewidth=1, color = 'red', ls = 'dashed')

if __name__ == '__main__':
    training_data = pd.read_csv('/Users/weiranlyu/Desktop/DecisionTree/bank/train.csv', header = None, dtype=str)
    training_data.columns = ['age','job','marital','education','default','balance','housing', 'loan',
              'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    testing_data = pd.read_csv('/Users/weiranlyu/Desktop/DecisionTree/bank/test.csv', header = None, dtype=str)
    testing_data.columns = training_data.columns

    #We want to convert numerical features to binary
    medAge = training_data['age'].median()
    training_data['age'] = (training_data['age'] >= str(medAge)).astype(int)

    medBalance = training_data['balance'].median()
    training_data['balance'] = (training_data['balance'] >= str(medBalance)).astype(int)

    medDay = training_data['day'].median()
    training_data['day'] = (training_data['day'] >= str(medDay)).astype(int)

    medDuration = training_data['duration'].median()
    training_data['duration'] = (training_data['duration'] >= str(medDuration)).astype(int)

    medCampaign = training_data['campaign'].median()
    training_data['campaign'] = (training_data['campaign'] >= str(medCampaign)).astype(int)

    medPdays = training_data['pdays'].median()
    training_data['pdays'] = (training_data['pdays'] >= str(medPdays)).astype(int)

    medPrevious = training_data['previous'].median()
    training_data['previous'] = (training_data['previous'] >= str(medPrevious)).astype(int)

    X_train, Y_train = training_data.ix[:,:-1], training_data.ix[:,-1]
    X_test, Y_test = testing_data.ix[:,:-1], testing_data.ix[:,-1]
    clf_tree = DT.ID3(training_data,training_data,training_data.columns[:-1], 1, 1)
    errorTree = clf(X_train, Y_train, X_test, Y_test, clf_tree)

    errorTrain, errorTest = [errorTree[0]], [errorTree[1]]
    T = range(1, 501)
    for i in T:    
        er_i = AdaBoost(X_train, Y_train, X_test, Y_test, i, clf_tree)
        errorTrain.append(er_i[0])
        errorTest.append(er_i[1])
    
    # The training and testing errors vary along with T
    plot_error_rate(errorTrain, errorTest)


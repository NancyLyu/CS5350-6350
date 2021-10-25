import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import DecisionTree as DT

def ID3(S, originalData, Attributes, Label = "y", commonLabel = None):
    #S = the dataset for which the ID3 algorithm should be run
    #Attributes = the set of the measured attributes
    #Label = the target attribute 'label'
    #max_depth = the maximum tree depth set by the user

    #if all examples have the same label, return a leaf node with the label
    if len(np.unique(S[Label])) <= 1:
        return np.unique(S[Label])[0]
    elif len(S)==0:
        return np.unique(originalData[Label])[np.argmax(np.unique(originalData[Label],return_counts=True)[1])]
     #if attributes is empty, return a leaf node with the most common label
    elif len(Attributes) == 0:
        return commonLabel
    else:
        commonLabel = np.unique(S[Label])[np.argmax(np.unique(S[Label],return_counts=True)[1])]

        #randomly select a subset of features. Vary the size of the feature subset from {2. 4, 6}
        features = random.choices(Attributes, k = 4)
        #print(features)
        igValues = [entropy_IG(S,att,Label) for att in features]

        bestSplit = np.argmax(igValues)
        bestAtt = features[bestSplit]
        #get the tree structure using dictionary
        tree = {bestAtt:{}}
        newAtts = [i for i in Attributes if i != bestAtt]
        for value in np.unique(S[bestAtt]):
            #Split the dataset and create S_v
            S_v = S.where(S[bestAtt] == value).dropna()
            #add subtree
            subtree = ID3(S_v,S,newAtts, Label, commonLabel)
            tree[bestAtt][value] = subtree
        return(tree)

def entropy_IG(S, attributes, Label = "y"):
    current_entropy = Entropy(S[Label])
    vals,counts= np.unique(S[attributes],return_counts=True)
    expected_Entropy = np.sum([(counts[i]/np.sum(counts))*Entropy(S.where(S[attributes]==vals[i]).dropna()[Label]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = current_entropy - expected_Entropy
    return Information_Gain

def Entropy(LabelValues):
    elements,counts = np.unique(LabelValues,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

training_data = pd.read_csv('/Users/weiranlyu/Desktop/DecisionTree/bank/train.csv', header = None, dtype=str)
training_data.columns = ['age','job','marital','education','default','balance','housing', 'loan',
              'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
testing_data = pd.read_csv('/Users/weiranlyu/Desktop/DecisionTree/bank/test.csv', header = None, dtype=str)
testing_data.columns = training_data.columns

#We want to convert numerical features to binary
medAge = training_data['age'].median()
training_data['age'] = (training_data['age'] >= str(medAge)).astype(int)

medBalance = training_data['balance'].median()
training_data['balance'] = (training_data['balance'] >= str(medAge)).astype(int)

medDay = training_data['day'].median()
training_data['day'] = (training_data['day'] >= str(medAge)).astype(int)

medDuration = training_data['duration'].median()
training_data['duration'] = (training_data['duration'] >= str(medAge)).astype(int)

medCampaign = training_data['campaign'].median()
training_data['campaign'] = (training_data['campaign'] >= str(medAge)).astype(int)

medPdays = training_data['pdays'].median()
training_data['pdays'] = (training_data['pdays'] >= str(medAge)).astype(int)

medPrevious = training_data['previous'].median()
training_data['previous'] = (training_data['previous'] >= str(medAge)).astype(int)

X_train, Y_train = training_data.ix[:,:-1], training_data.ix[:,-1]
X_test, Y_test = testing_data.ix[:,:-1], testing_data.ix[:,-1]

def RandomForest(data,number_of_Trees):
    #Create a list in which the single forests are stored
    RFSubtree = []
    size = data.size
    for i in range(number_of_Trees+1):
        #samples = data[np.random.choice(data.shape[0], int(size), replace = True)]
        samples = data.sample(frac=1,replace=True)
        RFSubtree.append([])
        RFSubtree[i] = ID3(samples,samples,training_data.columns[:-1])
    return RFSubtree

def RandomForest_Predict(data,forest):
    forest_size = len(forest)
    samples = len(data)
    tree_classification = np.zeros((samples, forest_size))
    ##With each tree, find the classification of each validation sample.
    predictions = []
    for tree in forest:
        predictions.append(DT.predicted(data,tree))
    return predictions

def RandomForest_Test(data,random_forest):
    data['predTrain'] = None
    data['predTest'] = None
    for i in range(len(data)):
        data.loc[i,'predTrain'] = RandomForest_Predict(X_train,random_forest)
        data.loc[i,'predTest'] = RandomForest_Predict(X_test,random_forest)
    errorTrain = sum(data['predTrain'] != data['y'])/len(data)*100
    errorTest = sum(data['predTest'] != data['y'])/len(data)*100
    return errorTrain, errorTest

def plot_error_rate(errorTrain, errorTest):
    df_error = pd.DataFrame([errorTrain, errorTest]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of random trees', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Random forest: error rate vs number of random trees', fontsize = 16)
    plt.axhline(y=errorTest[0], linewidth=1, color = 'red', ls = 'dashed')

T = range(1, 3)
errTrain, errTest = [], []
for i in T:
    rf = RandomForest(training_data,i)
    error = RandomForest_Test(training_data,rf)
    errTrain.append(error[0])
    errTest.append(error[1])

plot_error_rate(errTrain, errTest)



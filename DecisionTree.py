import pandas as pd
import numpy as np
import random


def ID3(S, originalData, Attributes, max_depth, depth, Label = "y", commonLabel = None):
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

        igValues = [entropy_IG(S,att,Label) for att in Attributes]

        bestSplit = np.argmax(igValues)
        bestAtt = Attributes[bestSplit]
        if max_depth < depth:
           return
        #get the tree structure using dictionary
        tree = {bestAtt:{}}
        newAtts = [i for i in Attributes if i != bestAtt]
        for value in np.unique(S[bestAtt]):
            #Split the dataset and create S_v
            S_v = S.where(S[bestAtt] == value).dropna()
            #add subtree
            subtree = ID3(S_v,S,newAtts,max_depth, depth + 1, Label, commonLabel)
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

def predict(temp, tree):
    for key in list(temp.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][temp[key]] 
            except:
                return 1 #the default value set as 1
            result = tree[key][temp[key]]
            #4.
            if isinstance(result,dict):
                return predict(temp,result)
            else:
                return result

def predicted(data, tree):
    temps = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(temps[i],tree)
    return predicted["predicted"]

def test(data,tree):
    temps = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(temps[i],tree) 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["y"])/len(data))*100,'%')

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

max_depth = 1
depth  = 1
tree = ID3(training_data,training_data,training_data.columns[:-1], max_depth, depth)
test(testing_data,tree)
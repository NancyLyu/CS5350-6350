import math
import csv
import numpy as np 

    
def ID3(S, Attributes, Label, attribute_selection, max_depth):
    depth = 0
    break_out_flag = False
    labels = S[Label].tolist()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if(labels[i] != labels[j]):
                break_out_flag = True
                break
        if break_out_flag:
            break
    if break_out_flag:
        if depth >= max_depth:
            return
        root, IG= {}, {}
        for att in Attributes:
            if attribute_selection == 0:
                IG[att]=entropy_IG(S, att, Label)
            if attribute_selection == 1:
                IG[att]=ME_IG(S, att, Label)
            if attribute_selection == 2:
                IG[att]=GI_IG(S, att, Label)
        MaxIG = max(IG, key=lambda key: IG[key]) #get the attribute with the max IG
        new_S = {}
        for S_key in S.groupby(Label).groups.keys():
            new_S[S_key] = S.groupby(Label).get_group(S_key)
        root = {MaxIG:{}}
        for v in new_S.keys(): #for each possible value v of that maxIG attribute can take
            v_branch = S.where(S[MaxIG] == v).dropna()
            if v_branch.empty:
                root[Max_IG][v] = new_S[v][Label].value_counts().idxmax()
            else:
                depth+1
                root[Max_IG][v] = ID3(v_branch, Attributes.remove(MaxIG), Label)
        return root
    else:
        if Attributes.empty:
            return S[Label].value_counts().idmax()
        return S[Label].tolist()[0]
        

def get_entropy(Label):
    e = 0
    unique,counts = np.unique(Label, return_counts = True)
    for c in counts:
        p = c/np.sum(counts)
        e += -p*np.log2(p)
    return e

def entropy_IG(S, att, Label):
    current_entropy = get_entropy(Label)
    expected_entropy = 0
    unique,counts = np.unique(S[att], return_counts = True)
    for i in range(len(counts)):
        p = counts[i]/np.sum(counts)
        new_S = S.where(S[att] == unique[i]).dropna()
        sub_labels = new_S[Label]
        expected_entropy += p*get_entropy(sub_labels)
    return current_entropy - expected_entropy

def major_error(Label):
    unique,counts = np.unique(Label, return_counts = True)
    p = []
    for c in counts:
        p.append(c/np.sum(counts))
    return min(p)

def ME_IG(S, att, Label):
    current_ME = major_error(Label)
    expected_ME = 0
    unique,counts = np.unique(S[att], return_counts = True)
    for i in range(len(counts)):
        p = counts[i]/np.sum(counts)
        new_S = S.where(S[att] == unique[i]).dropna()
        sub_labels = new_S[Label]
        expected_ME += p*major_error(sub_labels)
    return current_ME - expected_ME

def gini_index(Label):
    g = 0
    unique,counts = np.unique(Label, return_counts = True)
    for c in counts:
        p = c/np.sum(counts)
        g += p^2
    return 1 - g

def GI_IG(S, att, Label):
    current_GI = gini_index(Label)
    expected_GI = 0
    unique,counts = np.unique(S[att], return_counts = True)
    for i in range(len(counts)):
        p = counts[i]/np.sum(counts)
        new_S = S.where(S[att] == unique[i]).dropna()
        sub_labels = new_S[Label]
        expected_GI += p*gini_index(sub_labels)
    return current_GI - expected_GI

df = pd.read_csv('/Users/weiranlyu/Desktop/DecisionTree/car/train.csv', dtype=str)
df.columns = ['buying','maint','doors','persons','lug_boot','safety','label']
max_depth = 1
attribute_selection = 0
dt = ID3(df, list(df.columns), df.columns[len(df.columns) - 1], attribute_selection, max_depth)

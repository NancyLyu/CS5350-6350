import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def SGD(X, y, lr=0.05, epoch=10, batch_size=1): 
    m, b = 0.5, 0.5 # initial parameters
    log, mse = [], [] # lists to store learning process
    
    for _ in range(epoch):
        
        indexes = np.random.randint(0, len(X), batch_size) # random sample
        
        Xs = np.take(X, indexes)
        ys = np.take(y, indexes)
        N = len(Xs)
        
        f = ys - (m*Xs + b)
        
        # Updating parameters m and b
        m -= lr * (-2 * Xs.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        
        log.append((m, b))
        #mse.append(mean_squared_error(y, m*X+b))        
    return m, b, log, mse

def BGD(alpha, x, y, numIterations):
    m = x.shape[0] # number of samples
    theta = np.ones(2)
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost function   
        gradient = np.dot(x_transpose, loss) / m         
        theta = theta - alpha * gradient  # update
    return theta

training_data = pd.read_csv('/Users/weiranlyu/Desktop/Ensemble Learning/concrete/train.csv', header = None, dtype=str)
training_data.columns = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr']
testing_data = pd.read_csv('/Users/weiranlyu/Desktop/Ensemble Learning/concrete/test.csv', header = None, dtype=str)
testing_data.columns = training_data.columns


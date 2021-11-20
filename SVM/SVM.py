import numpy as np
import pandas as pd
import scipy.optimize as opt

class SVM:
    def __init__(self):
        self.C = 1
        self.lr = 0.1
        self.epoch = 100
        self.a = 0.1
        self.gamma = 0.2

    def set_C(self, C):
        self.C = C
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    #stochastic sub-gradient descent for SVM
    def GSD(self, x, y):
        #initialize w0
        weights = np.zeros(len(x[0]))
        #number of examples
        N = len(x)
        index = np.arange(N)
        for t in range(self.epoch):
            #shuffle the training examples for each epoch
            np.random.shuffle(index)
            x = x[index,:]
            y = y[index]
            for i in range(N):
                current_weights = weights.copy()
                current_weights[len(current_weights) - 1] = 0
                if y[i]*np.sum(np.multiply(weights, x[i])) <= 1:
                    current_weights = current_weights - self.C*N*y[i]*x[i]
                #use this schedule of learning rate for 2(a) 
                lr = self.lr/(1 + self.lr/self.a*t)
                #use this schedule of learning rate for 2(b)
                #lr = self.lr/(1 + t)
                weights = weights  - lr*current_weights
        return weights
    
    #dual SVM
    def dual(self, x, y):
        N = len(x)
        bounds = [(0, self.C)]*N
        constraints = ({'type': 'eq', 'fun': lambda alpha: self.constraint(alpha, y)})
        a_0 = np.zeros(N)
        res = opt.minimize(lambda alpha: self.objective(alpha, x, y), a_0,  method='SLSQP', bounds=bounds,constraints=constraints, options={'disp': False})
        
        weights = np.sum(np.multiply(np.multiply(np.reshape(res.x,(-1,1)), np.reshape(y, (-1,1))), x), axis=0)
        index = np.where((res.x > 0) & (res.x < self.C))
        b =  np.mean(y[index] - np.matmul(x[index,:], np.reshape(weights, (-1,1))))
        weights.tolist().append(b)
        weights = np.array(weights)
        return weights

    def constraint(self, alpha, y):
        c = np.matmul(np.reshape(alpha,(1, -1)), np.reshape(y,(-1,1)))
        return c[0]

    def objective(self, alpha, x, y):
        l = 0
        l = l - np.sum(alpha)
        ayx = np.multiply(np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1))), x)
        l = l + 0.5 * np.sum(np.matmul(ayx, np.transpose(ayx)))
        return l

    #Gaussain kernel
    def kernel(self, x1, x2, gamma):
        m1 = np.tile(x1, (1, x2.shape[0]))
        m1 = np.reshape(m1, (-1,x1.shape[1]))
        m2 = np.tile(x2, (x1.shape[0], 1))
        k = np.exp((np.sum(np.square(m1 - m2),axis=1) / -gamma).astype('float'))
        k = np.reshape(k, (x1.shape[0], x2.shape[0]))
        return k

    def objectiveGK(self, alpha, k, y):
        l = 0
        l = l - np.sum(alpha)
        ay = np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1)))
        ayay = np.matmul(ay, np.transpose(ay))
        l = l + 0.5 * np.sum(np.multiply(ayay, k))
        return l
    
    def Gaussian(self, x, y):
        N = x.shape[0]
        bounds = [(0, self.C)]*N
        constraints = ({'type': 'eq', 'fun': lambda alpha: self.constraint(alpha, y)})
        a_0 = np.zeros(N)
        k = self.kernel(x, x, self.gamma)
        res = opt.minimize(lambda alpha: self.objectiveGK(alpha, k, y), a_0,  method='SLSQP', bounds=bounds,constraints=constraints, options={'disp': False})
        return res.x
    
    def predictGaussian(self, alpha, x0, y0, x):
        k = self.kernel(x0, x, self.gamma)
        k = np.multiply(np.reshape(y0, (-1,1)), k)
        y = np.sum(np.multiply(np.reshape(alpha, (-1,1)), k), axis=0)
        y = np.reshape(y, (-1,1))
        y[y > 0] = 1
        y[y <= 0] = -1
        return y

    def predict(self, x, weights):
        prediction = np.matmul(x, weights)
        prediction[prediction > 0 ] = 1
        prediction[prediction <= 0] = -1
        return prediction

    def predictError(self, prediction, y):
        diff = prediction - np.reshape(y,(-1,1))
        error = np.sum(np.abs(diff))/2 
        return error/y.shape[0]

    

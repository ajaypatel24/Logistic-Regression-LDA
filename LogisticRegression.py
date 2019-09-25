import numpy as np 
import pandas as pd 
import matplotlib as mat 
from math import exp

class LogisticRegression: 

    def __init__(self, Input, Output, LR, GradientDescents, Weight):
        self.Input = input
        self.Output = Output
        self.LR = LR
        self.GradientDescents = GradientDescents
        self.Weight = Weight

    def sigmoid(self, prediction):
        return 1 / (1 + np.exp ((-(prediction))))



    def fit(x):
        print(x)

#feature = column 
#weight = importance
#play with weights to get bset possible

    def predict(self, features, weights):
        prediction = np.dot(features, weights)
        u = self.sigmoid(prediction)
        if (prediction >= 0.5):
            return 1
        else: 
            return 0

    def updateWeight(self, weights, input, output):

        weights = np.array(weights)
        
        h = np.multiply(input, output - self.sigmoid(np.dot(weights.T, input)))
        return h


    def evaluate_acc(self, x,y,z):
        print(x,y,z)

    

    
obj = LogisticRegression(1,2,3,4,5)

q = obj.updateWeight([3,5,6,2,8], [3,4,53,6,7], 1.0)
t = obj.predict([1,3,4,-43], [1,3,4,2])
print(t)
print(np.transpose(q))
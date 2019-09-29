import numpy as np 
import pandas as pd 
import matplotlib as mat 
from math import exp
import random
from scipy.special import expit

class LogisticRegression: 

    def __init__(self, Input, Output, LR, GradientDescents, Weight): #input (vector) output (vector) LR (Constant, alpha in the equation, speed of adjustment of weights) GradientDescents (constant,iterations basically) weight (input constant, becomes vector)
        self.Input = Input
        self.Output = Output
        self.LR = LR
        self.GradientDescents = GradientDescents
        self.Weight = Weight
        self.NumberInputs = Input.shape[0]

    def sigmoid(self, prediction):
        return expit(prediction) #avoid overflow errors

    def fit(self, input, output, LR, iterations): #train
        w = [self.Weight]
        for x in range (0,len(self.Input.iloc[0,:])-1):
            w.append(self.Weight) #create array of weights 

        for x in range (0, iterations):
            w = self.updateWeight(w, self.Input, self.Output)

        print(w)
        #w = [710, -300, 179, 291, -230, -400, -10, 400, -147, 270, 1745]
        result = []
        test = 1000
        for x in range(0,test):
            r = self.predict(self.Input.iloc[x,:], w)
            if (r > 0.5):
                result.append(1)
            else:
                result.append(0)
        print("result", result) 
        print(self.Output)
        self.evaluate_acc(result, self.Output.iloc[0:x])

    def predict(self, features, weights):
        prediction = np.dot(features, weights)
        return self.sigmoid(prediction)

    def updateWeight(self, weights, input, output): #gradient descent
        weights = np.array(weights)
        sum = [0.0] * np.size(weights)
        for x in range(0,len(input.iloc[:,1])):
            h = np.multiply(input.iloc[x,:], np.subtract(output.iloc[x],self.sigmoid(np.dot(weights.T, input.iloc[x,:]))))
            sum = np.add(sum,h)
        print("Sum", sum)
        updated = weights + np.multiply(self.LR, sum)
        print(updated)
        return updated
       
    def W(self, training, resultTraining):
        w = [self.Weight]
        for x in range (0,len(self.Input.iloc[0,:])-1):
            w.append(self.Weight) #create array of weights 
        for y in range (0, self.GradientDescents):
            w = self.updateWeight(w, training, resultTraining)
        return w

    def divideDataset(self, fold):
        # set = df.iloc[:,:-1] #all columns except for result 
        # output = df.iloc[:,-1] #output 
        rows = len(self.Input.iloc[:,1])
        Division = int(rows/fold)
        ws = []
        test = []
        accs = []
        for x in range (0, rows, Division):
            trainingSet = self.Input.drop(self.Input.index[x:x+Division])
            resultTraining = self.Output.drop(self.Output.index[x:x+Division])

            testSetInput = self.Input.iloc[x:x+Division]
            testSetOutput = self.Output.iloc[x:x+Division]

            w = self.W(trainingSet, resultTraining)
            accuracy = self.acc(w, testSetInput, testSetOutput)

            ws.append(w)
            accs.append(accuracy)

        print("weights",  ws)
        print("array", accs)
        return [ws, accs]

    def acc(self, w, input, output):
        correct = 0
        for x in range(0, len(input.iloc[:,1])):
            result = 0
            prediction = self.predict(input.iloc[x,:], w)
            if (prediction > 0.5):
                result = 1
            else:
                result = 0
            
            if (result == output.iloc[x]):
                correct += 1
        return correct / (len(input.iloc[:,1])) * 100

    def evaluate_acc(self, result, expected):
        c = 0
        for x in range (0,len(result)-1):
            if(result[x] == expected.iloc[x]):
                c += 1
        print(c / (c + len(result)) * 100)
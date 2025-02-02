import numpy as np 
import pandas as pd 
import matplotlib as mat 
from math import exp
import random
from scipy.special import expit

class LogisticRegression: 

    def __init__(self, data, LR, GradientDescents): 
        self.Input = data.iloc[:,:-1] #all columns except output
        self.Output = data.iloc[:,-1] #output column
        self.LR = LR #learning rate
        self.GradientDescents = GradientDescents 
        self.Weight = 0

    def sigmoid(self, prediction):
        return expit(prediction) #avoid overflow errors

    def gradientDescent(self, weights, input, output): #gradient descent
        weights = np.array(weights)
        sum = [0.0] * len(weights)
    
        for x in range(0,len(input.iloc[:,1])):
            CalculationGradient = np.multiply(input.iloc[x,:], (np.subtract(output.iloc[x], self.sigmoid(np.dot(weights.T, input.iloc[x,:])))))
            sum = np.add(sum,CalculationGradient)

        weights = weights + np.multiply(self.LR, sum)
        return weights
       
    def W(self, training, resultTraining): #updates weights according to training and test set
        w = [self.Weight]
        for x in range (0,len(self.Input.columns)-1): #create array of weights
            w.append(self.Weight) 

        for y in range (0, self.GradientDescents): 
            w = self.gradientDescent(w, training, resultTraining)
        return w

    def fit(self, input, output, LR, iterations): #train using all helper methods
        weights = self.W(input, output)
        result = []

        test = 240
        for x in range(0,test):
            r = self.predict(self.Input.iloc[x,:], weights)

            if (r > 0.5):
                result.append(1)
            else:
                result.append(0)

        accuracy = self.evaluate_acc(weights, result, output)
        print("acc", accuracy)
        print("result", result) 
        print(self.Output)
        self.evaluate_acc(result, self.Output.iloc[0:x])

    def predict(self, features, weights):
        prediction = np.dot(features, weights)
        return self.sigmoid(prediction)

    def addInteractionTerm(self): #TASK 3

        CitricPH = []
        input = self.Input
        CitricAcid = input["citric acid"].values
        PH = input["pH"].values
        
        for x in range(0, input.shape[0]):
            interact = CitricAcid[x] * PH[x]
            CitricPH.append(interact)

        input['CitricPH'] = CitricPH

        self.input = input

    def crossValidation(self, fold):
       
        rows = len(self.Input.iloc[:,1]) #total rows 
        Division = int(rows/fold) #depending on folds
        FinalWeights = []
        AccuracyArray = []

        for x in range (0, rows, Division):
            trainingSet = self.Input.drop(self.Input.index[x:x+Division]) #training set, varies according to fold 
            resultTraining = self.Output.drop(self.Output.index[x:x+Division])

            # print("Size of training set: ", trainingSet.shape[0])
            # print("Size of result set: ", resultTraining.shape[0])

            testSetInput = self.Input.iloc[x:x+Division] #held out test set
            testSetOutput = self.Output.iloc[x:x+Division]

            # print("Size of test set input: ", testSetInput.shape[0])
            # print("Size of test set output: ", testSetOutput.shape[0])
            w = self.W(trainingSet, resultTraining) #use training set to get weights
            
            accuracy = self.evaluate_acc(w, testSetInput, testSetOutput) #use weights to get prediction

            FinalWeights.append(w)
            AccuracyArray.append(accuracy)

        #print("weights",  FinalWeights)
        print("Accuracy over all 5 folds: ")
        print(AccuracyArray)
        sum = 0
        for x in (0,len(AccuracyArray)-1):  
            sum += x
        print ("Average accuracy: ", sum/5)

    def evaluate_acc(self, w, input, output):
        correct = 0
        for x in range(0, len(input.iloc[:,1])):
            result = 0
            r = []
            prediction = self.predict(input.iloc[x,:], w)
            if (prediction > 0.5):
                result = 1
            else:
                result = 0
            if (result == output.iloc[x]):
                # print("p", result)
                # print("p2", output.iloc[x])
                correct += 1

        print("correct", correct)
        print("len input", len(input.iloc[:,1]))
        return correct / (len(input.iloc[:,1])) * 100
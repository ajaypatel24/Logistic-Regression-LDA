import numpy as np 
import pandas as pd 
import matplotlib as mat 
from math import exp
import Wine


class LogisticRegression: 

    def __init__(self, Input, Output, LR, GradientDescents, Weight): #input (vector) output (vector) LR (Constant) GradientDescents (constant,iterations basically) weight (input constant, becomes vector)
        self.Input = Input
        self.Output = Output
        self.LR = LR
        self.GradientDescents = GradientDescents
        self.Weight = Weight
        self.NumberInputs = Input.shape[0]


    def sigmoid(self, prediction):
        prediction = int(prediction)
        #return 1 / (1 + np.exp ((-(prediction))))
        print(prediction)


    def fit(self, input, output, LR, iterations):
        w = [self.Weight]
        print("fit",w)
        for x in range (0,len(self.Input.iloc[0,:])-1):
            w.append(self.Weight) #create array of weights 

        for x in range (0, iterations):
            print("we out here", w)
            w = self.updateWeight(w, self.Input, self.Output)



        print("dub", w)
        print(self.Input.iloc[0,:])
        

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

        print("update weiht", weights)
        weights = np.array(weights)
        print("weights np", weights)
        cur = [0.0] * np.size(weights)
        print("cur", cur)

        for x in range(0,len(input.iloc[:,1])):
            h = np.multiply(input.iloc[x,:], (output.iloc[x] - output.iloc[x])) #self.sigmoid(np.dot(weights.T, input.iloc[x,:]))))
            cur = np.add(cur,h)

        #print(self.sigmoid(np.dot(weights.T, input.iloc[x,:])))
        print("weight pre trans", weights)
        print("weight", weights.T)
        print("input iloc", input.iloc[x,:])
        #updated = weights + np.multiply(self.LR, cur)
        #print("up", updated)
        print("w:", weights)
        #print("np.mult:", np.multiply(self.LR, cur))

       


    def evaluate_acc(self, x,y,z):
        print(x,y,z)

class Wine:

   

    def wineBinary(self):
        data = pd.read_csv("winequality-red.csv", sep=';')
        counter = 0
        for i in data.iloc[:,-1]:
            if (float(i) > 6.0):
                data.iat[counter,-1] = 1
            else:
                data.iat[counter,-1] = 0

            counter+=1

        g = len(data.columns)
        c = 0

        return data



q = Wine()
data = q.wineBinary()

obj = LogisticRegression( data.iloc[:,:-1],data.iloc[:,-1],0.8,3,0)
'''
print("in", obj.Input)
print("out", obj.Output)
print("LR", obj.LR)
print("GD", obj.GradientDescents)
'''
obj.fit(obj.Input, obj.Output, obj.LR, obj.GradientDescents)
#print(len(data))

#q = obj.updateWeight(obj.Input, [1,2,3,4,5], [1,1,1,1,1])
#t = obj.predict([1,3,4,-43], [1,3,4,2])



#obj.fit(obj.Input, obj.Output, 0.1, 50)
#print(t)
#print(np.transpose(q))


print("line break")


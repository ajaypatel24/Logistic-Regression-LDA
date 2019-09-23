import numpy as np 
import matplotlib as mat
import pandas as pd


class CleanData:

    data = pd.read_csv("winequality-red.csv", sep=';')

    def wineBinary(d):
        counter = 0
        for i in d.iloc[:,-1]:
            if (float(i) > 6.0):
                d.iat[counter,-1] = 1
            else:
                d.iat[counter,-1] = 0

            counter+=1

        g = len(d.columns)
        c = 0

        return d



    q = wineBinary(data)
    print(data)
 

    #print(q.iloc[:,-1].mean()) #calculates mean of a column 
    #print(q.iloc[:,-1].std()) #calculates mean of a column 








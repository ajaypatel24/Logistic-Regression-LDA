import numpy as np 
import matplotlib as mat
import pandas as pd




data = pd.read_csv("winequality-red.csv", sep=';')

def wineBinary(data):
    for i in data.iloc[:,-1]:
        if i > 5:
            data.iat[i,-1] = 1
        else:
            data.iat[i,-1] = 0

    return data

q = wineBinary(data)
print(q.iloc[0,-1])








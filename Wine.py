import numpy as np 
import matplotlib as mat
import pandas as pd


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











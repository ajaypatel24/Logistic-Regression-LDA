import numpy as np
import pandas as pd
import DataPreprocessCancer as dpc
from LogisticRegression import LogisticRegression

class Experiments:

    CANCER_DATA_DIR = 'breast-cancer-wisconsin.data'
    WINE_DATA_DIR = 'winequality-red.csv'

    #open file and convert to matrix
    cancer_matrix = np.loadtxt(open(CANCER_DATA_DIR, "rb"), dtype='str', delimiter=",")

    #convert to pandas dataframe and start cleaning
    df = pd.DataFrame(cancer_matrix)
    df = dpc.cleanValues(df)
    df = dpc.strToInt(df)
    df = dpc.replaceBinaryValues(df)
    df = dpc.removeOutliers(df)

    #analysis
    dpc.statisticalAnalysis(df)

    #logistic regression
    lr = LogisticRegression(df.iloc[:,:-1],df.iloc[:,-1],0.1,100,0)
    lr.divideDataset(5)
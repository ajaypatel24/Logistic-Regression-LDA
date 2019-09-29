import numpy as np
import pandas as pd
import DataPreprocessCancer as dpc
import DataPreprocessWine as dpw
from Logistic import LogisticRegression

class Experiments:

    CANCER_DATA_DIR = 'breast-cancer-wisconsin.data'
    WINE_DATA_DIR = 'winequality-red.csv'

    #open cancer data file and convert to matrix
    cancer_matrix = np.loadtxt(open(CANCER_DATA_DIR, "rb"), dtype='str', delimiter=",")

    #convert to pandas dataframe and start cleaning
    cancer_df = pd.DataFrame(cancer_matrix)
    cancer_df = dpc.cleanValues(cancer_df)
    cancer_df = dpc.strToInt(cancer_df)
    cancer_df = dpc.replaceBinaryValues(cancer_df)
    cancer_df = dpc.removeOutliers(cancer_df)

    #import wine data and clean
    wine_df = pd.read_csv(WINE_DATA_DIR, sep=';')
    wine_df = dpw.wineBinary(wine_df)
    wine_df = dpw.removeOutliers(wine_df)

    #analysis
    dpc.statisticalAnalysis(cancer_df)

    #logistic regression on cancer
    lr = LogisticRegression(cancer_df.iloc[:,:-1],cancer_df.iloc[:,-1],0.1,100,0)
    #lr.divideDataset(5)
    lr.fit(lr.Input, lr.Output, lr.LR, lr.GradientDescents)

    #logistic regression on wine
import numpy as np
import pandas as pd
import DataPreprocessCancer as dpc

class Experiments:

    CANCER_DATA_DIR = 'breast-cancer-wisconsin.data'
    WINE_DATA_DIR = 'winequality-red.csv'

    #open file and convert to matrix
    cancer_matrix = np.loadtxt(open(CANCER_DATA_DIR, "rb"), dtype='str', delimiter=",")

    #convert to pandas dataframe and start cleaning
    df = pd.DataFrame(cancer_matrix)
    df = dpc.cleanValues(df)
    df = dpc.strToFloat(df)
    df = dpc.replaceBinaryValues(df)

    #analysis
    dpc.statisticalAnalysis(df)
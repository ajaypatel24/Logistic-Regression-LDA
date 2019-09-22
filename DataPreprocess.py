import numpy as np
import pandas as pd

#directories of the data
WINE_DATA_DIR = 'C:/Users/Lara/workspaceF19/comp551/project1/datasets/winequality-red.csv'
CANCER_DATA_DIR = 'C:/Users/Lara/workspaceF19/comp551/project1/datasets/breast-cancer-wisconsin.data'

#drop NaN and ? from matrix
def cleanValues(df):
    df.dropna()
    df = df[~(df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] == '?').any(axis=1)]
    return df

#convert from string to float
def strToFloat(df):
    col = df.shape[1]
    for i in range(col):
        df[i] = df[i].astype(float)
    return df

#display statistical analysis
def statisticalAnalysis(df):
    col = df.shape[1]
    print("Column mean max min median std")
    for i in range(col):
        print(str(i) + ' ' + str(df[i].mean()) + ' ' + str(df[i].max()) + ' ' + str(df[i].min()) + ' ' + str(df[i].median()) + ' ' + str(df[i].std()))

#replacing binary values by 0 and 1
def replaceBinaryValues(df):
    df[10].replace([2, 4], [0, 1], inplace=True)
    return df

#open file and convert to matrix
mat = np.loadtxt(open(CANCER_DATA_DIR, "rb"), dtype='str', delimiter=",")

#convert to pandas dataframe and start cleaning
df = pd.DataFrame(mat)
df = cleanValues(df)
df = strToFloat(df)
df = replaceBinaryValues(df)

#analysis
statisticalAnalysis(df)
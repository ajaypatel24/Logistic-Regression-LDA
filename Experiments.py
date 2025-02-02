import numpy as np
import pandas as pd

import DataPreprocessCancer as dpc
import DataPreprocessWine as dpw
import LinearDiscriminantAnalysis as lda
from Logistic import LogisticRegression

CANCER_DATA_DIR = 'breast-cancer-wisconsin.data'
WINE_DATA_DIR = 'winequality-red.csv'

print("BREAST CANCER DATASET --------------------------------------------------------------------------------------")

#open cancer data file and start cleaning
print("Cleaning the breast cancer dataset...")
cancer_matrix = np.loadtxt(open(CANCER_DATA_DIR, "rb"), dtype='str', delimiter=",")

cancer_df = pd.DataFrame(cancer_matrix)
pre_shape = cancer_df.shape
print("Removing rows containing non-numerical or empty values...")
cancer_df = dpc.cleanValues(cancer_df)
print("Converting to int, replacing last column with binary values...")
cancer_df = dpc.strToInt(cancer_df)
cancer_df = dpc.replaceBinaryValues(cancer_df)
print("Removing outliers from the dataset...")
cancer_df = dpc.removeOutliers(cancer_df)
final_shape = cancer_df.shape

#Statistics
print("Some statistics on the cancer dataset: ")
print("Shape before cleaning: " + str(pre_shape))
print("Shape after cleaning: " + str(final_shape))
dpc.statisticalAnalysis(cancer_df)
print("Percentage of malignant tumors: " + str(dpc.dataRatio(cancer_df)))

#logistic regression on cancer
print("Performing logistic regression on the cancer dataset -------------------------------------------------------------------")
lrc = LogisticRegression(cancer_df,0.11,100)
lrc.crossValidation(5)
lrc.fit(lrc.Input, lrc.Output, lrc.LR, lrc.GradientDescents)

#LDA on cancer
print("Performing linear discriminant analysis on the cancer dataset ----------------------------------------------------------")
results_c = lda.k_fold(cancer_df, 5)
avg = np.mean(results_c)
std = np.std(results_c)
print("The accuracy of each of the 5 folds are: ")
print(results_c)
print("The average accuracy over the 5 folds is: " + str(avg))
print("With a standard deviation of " + str(std))

print("WINE DATASET -----------------------------------------------------------------------------------------------------------")

#import wine data and clean
print("Cleaning the wine dataset...")
wine_df = pd.read_csv(WINE_DATA_DIR, sep=';')
pre_shape = wine_df.shape
print("Replacing last column with binary values...")
wine_df = dpw.wineBinary(wine_df)
print("Removing outliers from the dataset...")
wine_df = dpw.removeOutliers(wine_df)
final_shape = wine_df.shape

#Statistics
print("Some statistics on the wine dataset: ")
print("Shape before cleaning: " + str(pre_shape))
print("Shape after cleaning: " + str(final_shape))
dpw.statisticalAnalysis(wine_df)
print("Percentage of good wines: " + str(dpw.dataRatio(wine_df)))

#logistic regression on wine
lrw = LogisticRegression(wine_df,0.11,100)
lrw.crossValidation(5)

#LDA on wine
print("Performing linear discriminant analysis on the wine dataset ------------------------------------------------------------")
results_w = lda.k_fold(wine_df, 5)
avg = np.mean(results_w)
std = np.std(results_w)
print("The accuracy of each of the 5 folds are: ")
print(results_w)
print("The average accuracy over the 5 folds is: " + str(avg))
print("With a standard deviation of " + str(std))

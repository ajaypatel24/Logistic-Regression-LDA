#drop NaN and ? from matrix
def cleanValues(df):
    df = df.dropna()
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df = df[~(df[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] == '?').any(axis=1)]
    return df

#convert from string to int
def strToInt(df):
    col = df.shape[1] + 1
    for i in range(1,col):
        df[i] = df[i].astype(int)
    return df

#display statistical analysis
def statisticalAnalysis(df):
    col = df.shape[1] + 1
    print("Column mean max min median std")
    for i in range(1, col):
        print(str(i) + ' ' + str(df[i].mean()) + ' ' + str(df[i].max()) + ' ' + str(df[i].min()) + ' ' + str(df[i].median()) + ' ' + str(df[i].std()))

#replacing binary values by 0 and 1
def replaceBinaryValues(df):
    #df[10].replace([2, 4], [0, 1], inplace=True)
    count = 0
    for i in df.iloc[:,-1]:
            if (i < 3):
                df.iat[count,-1] = 0
            else:
                df.iat[count,-1] = 1

            count += 1
    return df

#removing outliers
def removeOutliers(df):
    for i in range(1,len(df.columns)-1):
        Q1 = df.iloc[:,i].quantile(0.25)
        Q3 = df.iloc[:,i].quantile(0.75)
        IQR = float("{0:.2f}".format(Q3-Q1))
        counter = 0
        for z in df.iloc[:,i].to_numpy():
            if (IQR == 0.0):
                break
            if (z > Q3 + (1.5 * IQR) or z < Q1 - (1.5 * IQR)  ):
                try: 
                    df.drop([counter], inplace=True)
                except:
                    continue
            counter+=1
    return df
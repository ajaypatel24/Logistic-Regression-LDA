class DataPreprocessCancer:

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
def wineBinary(data): #also drops all rows that contain statistical outliers according to Q1 Q3 and IQR
    counter = 0
    for i in data.iloc[:,-1]:
        if (float(i) > 5.0):
            data.iat[counter,-1] = 1
        else:
            data.iat[counter,-1] = 0
        counter += 1
    return data

def removeOutliers(data):
    for i in range(0,len(data.columns)-1):
        counter = 0
        #for x in range(0,len(data.iloc[:,i])):
        Q1 = data.iloc[:,i].quantile(0.25)
        Q3 = data.iloc[:,i].quantile(0.75)
        IQR = float("{0:.2f}".format(Q3-Q1))
        counter = 0
        for z in data.iloc[:,i].to_numpy():
            if (IQR == 0.0):
                #print("zero on 10")
                break
            if (z > Q3 + (1.5 * IQR) or z < Q1 - (1.5 * IQR)  ):
                try: 
                    data.drop([counter], inplace=True)
                except:
                    continue
            counter+=1
    return data

def statisticalAnalysis(df):
    col = df.shape[1] -1
    titles = [ "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality" ]
    print("For each of the features, here are some stats: ")
    for i in range(0, col):
        print(titles[i] + ' -> mean: ' + str(df.iloc[:,i].mean()) + ', median: ' + str(df.iloc[:,i].median()) + ', std: ' + str(df.iloc[:,i].std()) + ', max: ' + str(df.iloc[:,i].max()) + ', min: ' + str(df.iloc[:,i].min()))

def dataRatio(df):
    good = 0
    total = df.shape[0]
    for i in df.iloc[:,-1]:
        if i > 0:
            good+=1
    return good/total
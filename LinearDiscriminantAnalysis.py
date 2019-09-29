import numpy as np
import pandas as pd

class LinearDiscriminantAnalysis:

    def __init__(self):
        self.w1 = None
        self.w0 = None
    
    def fit(self, X_train, y_train):
        data = X_train
        data['quality'] = y_train
        
        u0 = data.loc[data['quality']==0].iloc[:, :len(data.columns)-1]
        u1 = data.loc[data['quality']==1].iloc[:, :len(data.columns)-1]
        
        first_class_count = float(len(u0))
        second_class_count = float(len(u1))
        
        first_class_prob = first_class_count/len(y_train)
        second_class_prob = second_class_count/len(y_train)

        first_average = []
        second_average = []
        for column in u0:
            column = u0[column].values
            first_average.append(np.mean(column))

        for column in u1:
            column = u1[column].values
            second_average.append(np.mean(column))

        first_average = np.transpose(np.expand_dims(first_average, axis=1))
        second_average = np.transpose(np.expand_dims(second_average, axis=1))
        
        cov0 = np.cov(np.transpose(u0))
        cov1 = np.cov(np.transpose(u1))
        covariance_matrix = cov1 + cov0
        
        division = second_class_prob/first_class_prob
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        class_two = 0.5*float(np.matmul(np.matmul(second_average,inv_covariance_matrix),np.transpose(second_average)))
        class_one = 0.5*float(np.matmul(np.matmul(first_average,inv_covariance_matrix),np.transpose(first_average)))
        
        self.w0 = np.log(division) - class_two + class_one
        self.w1 = np.matmul(inv_covariance_matrix, (np.transpose(second_average)-np.transpose(first_average)))
        
    def predict(self, X_test):
        X_test = X_test.to_numpy()
        return [1 if np.dot(x,self.w1) + self.w0 > 0 else 0 for x in X_test]

def k_fold(data, n):
    results = []
    fold_length = int(len(data)/5)
    for i in range(n):
        start = fold_length*i
        end = start + fold_length
        temp_data = data
        
        temp_data_test = temp_data.iloc[start:end]
        
        X_test = temp_data_test.iloc[:, :len(temp_data_test.columns)-1]
        y_test = temp_data_test.iloc[:, len(temp_data_test.columns)-1:].values
        
        top = temp_data.iloc[0:start]
        bottom = temp_data.iloc[end:len(temp_data)]
        
        temp_data_train = pd.concat([top,bottom])
        
        X_train = temp_data_test.iloc[:, :len(temp_data_train.columns)-1]
        y_train = temp_data_test.iloc[:, len(temp_data_train.columns)-1:].values
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        
        correct = 0.0
        for pred,true in zip(y_pred, y_test):
            if pred == true:
                correct += 1 
        results.append(correct/len(y_train))
    return results
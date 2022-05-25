# -*- coding: utf-8 -*-

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

def process(path):
        #importing the dataset
        dataset = pd.read_csv(path)
        
        x = dataset.iloc[ : , :-1].values
        y = dataset.iloc[:, -1:].values
        
        
        #spliting the dataset into training set and test set
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )

        classifier = XGBClassifier(random_state = 0)
        classifier.fit(x_train, y_train)

        #sorted_idx=np.argsort(classifier.feature_importances_)[::-1]
        #for index in sorted_idx:
                #print([x_train.columns[index], classifier.feature_importances_[index]])

        


        #predicting the tests set result
        y_pred=classifier.predict(x_test)


        result2=open("results/resultXGBClassifier.csv","w")
        result2.write("ID,Predicted Value" + "\n")
        for j in range(len(y_pred)):
            result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
        result2.close()
        
        mse=mean_squared_error(y_test, y_pred)
        mae=mean_absolute_error(y_test, y_pred)
        r2=r2_score(y_test, y_pred)
        conf_matrix=confusion_matrix(y_test, y_pred)
        print("Confusion matrix",confusion_matrix(y_test, y_pred))
        print("precision_score==",precision_score(y_test, y_pred))
        print("recall_score=",recall_score(y_test, y_pred))
        
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
         
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()
        print("---------------------------------------------------------")
        print("MSE VALUE FOR XGBClassifier IS %f "  % mse)
        print("MAE VALUE FOR XGBClassifier IS %f "  % mae)
        print("R-SQUARED VALUE FOR XGBClassifier IS %f "  % r2)
        rms = np.sqrt(mean_squared_error(y_test, y_pred))
        print("RMSE VALUE FOR XGBClassifier IS %f "  % rms)
        ac=accuracy_score(y_test,y_pred)
        print ("ACCURACY VALUE XGBClassifier IS %f" % ac)
        print("---------------------------------------------------------")
        

        result2=open('results/XGBClassifierMetrics.csv', 'w')
        result2.write("Parameter,Value" + "\n")
        result2.write("MSE" + "," +str(mse) + "\n")
        result2.write("MAE" + "," +str(mae) + "\n")
        result2.write("R-SQUARED" + "," +str(r2) + "\n")
        result2.write("RMSE" + "," +str(rms) + "\n")
        result2.write("ACCURACY" + "," +str(ac) + "\n")
        result2.close()


         
        
        df =  pd.read_csv('results/XGBClassifierMetrics.csv')
        acc = df["Value"]
        alc = df["Parameter"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
        explode = (0.1, 0, 0, 0, 0)  
        
        fig = plt.figure()
        plt.bar(alc, acc,color=colors)
        plt.xlabel('Parameter')
        plt.ylabel('Value')
        plt.title('XGBClassifier Metrics Value')
        fig.savefig('results/XGBClassifierMetricsValue.png') 
        plt.show()
        y_pred = classifier.predict(x_test)
        fig = plt.figure()
        plot_importance(classifier, max_num_features=10)
        plt.xlabel('F-score')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        fig.savefig('results/Feature_importance.png')
        plt.show()
        


         
        
       
#process("data.csv")

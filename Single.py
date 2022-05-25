import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import inputScript

def process(path,inputurl):

	#importing the dataset
	dataset = pd.read_csv(path)

	x = dataset.iloc[ : , :-1].values
	y = dataset.iloc[:, -1:].values

	#spliting the dataset into training set and test set
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )


	#fitting RandomForest regression with best params 
	classifier = LogisticRegression()
	classifier.fit(x_train, y_train)

	#predicting the tests set result
	y_pred = classifier.predict(x_test)

	#confusion matrix
	cm = confusion_matrix(y_test, y_pred)
	print(cm)


	#pickle file joblib
	joblib.dump(classifier, 'results/lr_final.pkl')



	result2=open("results/resultLR.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()
	
	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=r2_score(y_test, y_pred)
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR Logistic Regression IS %f "  % mse)
	print("MAE VALUE FOR Logistic Regression IS %f "  % mae)
	print("R-SQUARED VALUE FOR Logistic Regression IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	print("RMSE VALUE FOR Logistic Regression IS %f "  % rms)
	ac=accuracy_score(y_test,y_pred)
	print ("ACCURACY VALUE Logistic Regression IS %f" % ac)
	print("---------------------------------------------------------")
	

	result2=open('results/LRMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/LRMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title(' Logistic Regression Metrics Value')
	fig.savefig('results/LRMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	
	#load the pickle file
	classifier = joblib.load('results/lr_final.pkl')
	#checking and predicting
	checkprediction = inputScript.process(inputurl)
	prediction = classifier.predict(checkprediction)
	print(prediction)
	return prediction


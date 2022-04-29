# -*- coding: utf-8 -*-

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score




from keras.models import Sequential
from keras.layers import Dense, Activation ,Dropout , Flatten , Conv1D ,MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
from timeit import default_timer as timer


def build_model(input):
	model = Sequential()
	model.add(Dense(128,input_shape=(input[1],input[2])))
	model.add(Conv1D(filters = 112, kernel_size= 1,padding='valid', activation='relu', kernel_initializer="uniform"))
	model.add(MaxPooling1D(pool_size=2, padding='valid'))
	model.add(Conv1D(filters = 64,kernel_size = 1,padding='valid', activation='relu', kernel_initializer="uniform"))
	model.add(MaxPooling1D(pool_size=1, padding='valid'))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(100, activation="relu", kernel_initializer="uniform"))
	#model.add(Dropout(0.2))
	model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
	model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	return model

def process(path):
	#importing the dataset
	df = pd.read_csv(path)
	
	x = df.iloc[ : , :-1].values
	y = df.iloc[:, -1:].values
	
	#spliting the dataset into training set and test set
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )

	#amount_of_features = len(df.columns)
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1)) 

	model = build_model([x_train.shape[0], x_train.shape[1],1])
	#Summary of the Model
	print(model.summary())
	
	start = timer()
	hist = model.fit(x_train,y_train,batch_size=128,epochs=25,validation_split=0.2,verbose=2)
	
	model.save('results/CNN.h5');
	#model evaluation
	test_loss,test_acc=model.evaluate(x_test,y_test)
	print(test_loss,test_acc)

	#train and validation loss
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['train','Validation'],loc='upper left')
	plt.savefig('results/CNN Loss.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	#train and validation accuracy
	plt.plot(hist.history['accuracy'])
	plt.plot(hist.history['val_accuracy'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train','Validation'],loc='upper left')
	plt.savefig('results/CNN Accuracy.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	

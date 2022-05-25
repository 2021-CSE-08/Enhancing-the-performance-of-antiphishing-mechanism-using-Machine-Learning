import numpy as np
import pandas as pd


from keras.models import Sequential,load_model
from keras.layers import Dense, Activation ,Dropout , Flatten , Conv1D ,MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
from timeit import default_timer as timer
import inputScript

def CNNLSTMbuild_model(input):
	model = Sequential()
	model.add(Dense(128,input_shape=(input[1],input[2])))
	model.add(Conv1D(filters = 24, kernel_size= 1,padding='valid', activation='relu', kernel_initializer="uniform"))
	model.add(MaxPooling1D(pool_size=2, padding='valid'))
	model.add(Conv1D(filters = 48,kernel_size = 1,padding='valid', activation='relu', kernel_initializer="uniform"))
	model.add(MaxPooling1D(pool_size=2, padding='valid'))
	model.add(LSTM(40,return_sequences=True))
	model.add(LSTM(32,return_sequences=False))
	model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
	#model.add(Dropout(0.2))
	model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
	model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	return model

def process(path,inputurl):
	model = load_model('results/CNNLSTM.h5')
	df = pd.read_csv(path)
	
	x = df.iloc[ : , :-1].values
	y = df.iloc[:, -1:].values
	
	classifier = CNNLSTMbuild_model([x.shape[0], x.shape[1],1])


	#test_X=[]

	checkprediction = inputScript.process(inputurl)
	x_test = np.array(checkprediction)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1)) 
	print(x_test)
	print(x_test.shape)
	prediction = classifier.predict(x_test)
	print(prediction[0][0])
	result=""
	if prediction[0][0] > 0:
		result=1
	else:
		result=-1
	print(result)
	return result

#process("data.csv","http://ssstrades.com/Chase/chase")


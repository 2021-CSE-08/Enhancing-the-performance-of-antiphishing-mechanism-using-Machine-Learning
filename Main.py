import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import preprocess as pre
import CNN as CN
import XGBoost as XGB 
#from sklearn.externals import joblib
import Single as Sin
import inputScript
import Predict as pr

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"


def Home():
	global window
	def clear():
	    print("Clear1")
	    txt.delete(0, 'end')
	    txt1.delete(0, 'end')



	window = tk.Tk()
	window.title("Detecting Phising Website")

 
	window.geometry('1280x720')
	window.configure(background=bgcolor)
	#window.attributes('-fullscreen', True)

	window.grid_rowconfigure(0, weight=1)
	window.grid_columnconfigure(0, weight=1)
	

	message1 = tk.Label(window, text="Detecting Phising Website" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
	message1.place(x=100, y=20)

	lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl.place(x=100, y=200)
	
	txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt.place(x=400, y=215)

	lbl1 = tk.Label(window, text="Enter URL",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl1.place(x=100, y=300)
	
	txt1 = tk.Entry(window,width=40,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt1.place(x=400, y=315)

	def browse():
		path=filedialog.askopenfilename()
		print(path)
		txt.insert('end',path)
		if path !="":
			print(path)
		else:
			tm.showinfo("Input error", "Select Train Dataset")	

	def preproc():
		sym=txt.get()
		if sym != "" :
			pre.process(sym)
			print("preprocess")
			tm.showinfo("Input", "Preprocess Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")

	def CNNprocess():
		sym=txt.get()
		if sym != "" :
			CN.process(sym)
			tm.showinfo("Input", "CNN Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")

	
	

	def XGBprocess():
		sym=txt.get()
		if sym != "" :
			XGB.process(sym)
			tm.showinfo("Input", "XGBoost Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")

	def Predicted():
		sym=txt.get()
		sym1=txt1.get()

		if sym != "" and sym1!="" :
			prediction = pr.process(sym,sym1)
			if prediction==1:
				tm.showinfo("Result", "Predicted Class is Genuine URL ")
				print(prediction)
			else:
				tm.showinfo("Result", "Predicted Class is Phishing URL ")
						       
			
		else:
			tm.showinfo("Input error", "Select Dataset and Enter Url")

	browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	browse.place(x=650, y=200)

	clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
	clearButton.place(x=950, y=200)
	 
	proc = tk.Button(window, text="Preprocess", command=preproc  ,fg=fgcolor   ,bg=bgcolor1   ,width=16  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	proc.place(x=50, y=600)
	
	LRbutton = tk.Button(window, text="CNN", command=CNNprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=16  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	LRbutton.place(x=250, y=600)

##	NNMbutton = tk.Button(window, text="CNNLSTM", command=CNNLSTMprocess  ,fg=fgcolor   ,bg=bgcolor1 ,width=16  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
##	NNMbutton.place(x=450, y=600)

	Xbutton = tk.Button(window, text="XGBoost", command=XGBprocess  ,fg=fgcolor   ,bg=bgcolor1 ,width=16  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	Xbutton.place(x=450, y=600)

	PRbutton = tk.Button(window, text="Prediction", command=Predicted  ,fg=fgcolor   ,bg=bgcolor1   ,width=16  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	PRbutton.place(x=650, y=600)


	quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	quitWindow.place(x=850, y=600)

	window.mainloop()
Home()


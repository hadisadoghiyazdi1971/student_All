import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import time # For Save Same Files Without Overlap

Symbol = "Synthesis"

F = open(os.getcwd() + "\\" + Symbol + "_Predictions.txt", "rb")
#F = open(os.getcwd() + "\\Results\\1672208111\\" + Symbol + "_Predictions.txt", "rb")
#F = open(os.getcwd() + "\\Results\\1672208111\\Editted_" + Symbol + "_Predictions.txt", "rb")
Temp = pickle.load(F)
F.close()

Methods = Temp["Methods"]
Y_Prediction = Temp["Predictions"]

Smoothing_WindowSize = 3
Y = np.expand_dims(np.tile(np.abs(np.arange(-23, 23)), 5), axis = 0)
##Synthesis_Data = True
#Y = np.add(Y, np.random.normal(0, 1.3, size = Y.shape ))
##Y_Smooth = np.convolve(Y[0], np.ones(Smoothing_WindowSize) / Smoothing_WindowSize, "same")
##Y_Smooth = np.expand_dims(Y_Smooth, axis = 0)
##Symbol = "Synthesis"

##Loss_Indices = [24, 47, 70, 93]
##Peaks_Indices = [47]
##Valeys_Indices = [116]


##Critical_Point = [23 * i for i in range(1, int(len(Y[0]) / 23) + 1)]
##Critical_Point[-1] = Critical_Point[-1] - 1
##Critical_Point_Type = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1] # +1: Valey, -1: Peak

Z = np.array([[1, 0]])
T = np.array([[1, 1], [0, 1]])
H = np.array([[np.random.rand() * 10]])
R = np.eye(2)
Q = np.random.rand(2,2)

Y_Plot = np.array([])
Label_Plot = []

Iteration = 100
for i in range(int(Methods.shape[0] / Iteration)):
  Temp = Y_Prediction[(i * Iteration):((i + 1) * Iteration), :]
  Temp = np.mean(Temp, axis = 0)
  if len(Temp.shape) < 2: Temp = np.expand_dims(Temp, axis = 0)
  Y_Plot = np.concatenate((Y_Plot, Temp), axis = 0) if len(Y_Plot.shape) > 1 else Temp
  Label_Plot += [Methods[i * Iteration]]

for i in range(len(Label_Plot)):
  if "ARIMA" in Label_Plot[i] or Label_Plot[i][0] =="(":
    Label_Plot[i] = Label_Plot[i][:Label_Plot[i].rfind(")") + 1]
  elif "PseudoPinBall_" in Label_Plot[i]:
    Temp = Label_Plot[i].split("_")
    Label_Plot[i] = "_".join(Temp[:2]) + r" ($\tau = {}$".format(float(Temp[3])) + ")"
  elif "SGD_PinBall_" in Label_Plot[i]:
    Temp = Label_Plot[i].split("_")
    Label_Plot[i] = "_".join(Temp[:3]) + r" ($\tau = {}$".format(float(Temp[4])) + r", $\mu = {}$".format(float(Temp[7])) + ")"
  elif "LSTM_" in Label_Plot[i]:
    Temp =  Label_Plot[i].split("_")
    Label_Plot[i] = Temp[0] + " ($ N = {}$".format(int(Temp[1].replace("N", ""))) + ", $W = {}$".format(int(Temp[2].replace("W", ""))) + ")"
  elif "CorrEntropy_" in Label_Plot[i]:
    Temp = Label_Plot[i].split("_")
    Label_Plot[i] = "_".join(Temp[:2]) + r" ( $\sigma = {}$".format(float(Temp[3])) + ")"
  elif "Huber_" in Label_Plot[i]:
    Temp = Label_Plot[i].split("_")
    Label_Plot[i] = "_".join(Temp[:2]) + r" ( $\delta = {}$".format(float(Temp[3])) + ")"
  Label_Plot[i] = r"{}".format(Label_Plot[i])

Label_Plot = np.array(Label_Plot)

for i in range(len(Label_Plot)):
  print(Label_Plot[i])


Method_Indices = [0, 1, 2, 3, 4, 7, 28, 31, 32, 35, 36, 37, 45] # 
Plot_Indices = list(range(Y.shape[1]))
#Plot_Indices = list(range(35, 59))

Line_Widths = [3.1, 3.1, 2.3, 2.3, 1.9, 1.9, 1.9, 1.7, 1.7, 1.7, 1.7, 1.3, 2.3]
Line_Styles = ["--", "-.", "--", "-.", "--", "--", "--", "--", "--", "--", "--", "--", "--"]

plt.figure(figsize = (17, 8))
plt.title(Symbol)
plt.plot(Plot_Indices, Y[0, Plot_Indices], label = "Input", linewidth = 5.3)
for i in range(len(Method_Indices)):
  plt.plot(Plot_Indices, Y_Plot[Method_Indices[i], Plot_Indices], label = Label_Plot[Method_Indices[i]], linestyle = Line_Styles[i], linewidth = Line_Widths[i])
  """
  if "SGD" in Label_Plot[Method_Indices[i]]:
    plt.plot(Plot_Indices, Y_Plot[Method_Indices[i], Plot_Indices], label = Label_Plot[Method_Indices[i]], linestyle = "--", linewidth = 2.3)
  else:
    plt.plot(Plot_Indices, Y_Plot[Method_Indices[i], Plot_Indices], label = Label_Plot[Method_Indices[i]], linestyle = "--")
  """

plt.legend()
plt.tight_layout()
#plt.savefig(os.getcwd() + "\\Results\\1672208111\\"+ Symbol + "_Plot_.png")
#plt.savefig(os.getcwd() + "\\Results\\1672208111\\Zoom_"+ Symbol + "_Plot.png")
plt.show()

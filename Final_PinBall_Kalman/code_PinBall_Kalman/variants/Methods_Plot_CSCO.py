import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import time # For Save Same Files Without Overlap

Symbol = "CSCO"

Width_Coefficient = 0.03 # Difference From Center
Height_Coefficient = 0.07

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


#["ARIMA(5, 0, 0)", "ARIMA(10, 0, 0)", "ARIMA(5, 1, 10)", "ARIMA(10, 1, 10)"
# , "KF_MSE", "KF_CorrEntropy", "KF_PseudoHuber", "LSTM(15, 5)"
# , "LSTM(20, 5)", "LSTM(15, 10)", "LSTM(20, 10)"
# , "KF_PseudoPinBall", "KF_SGD_PinBall"]
#
#Method_Indices = [0, 1, 2, 3, 4, 7, 28, 31, 32, 35, 36, 37, 45] #
#Line_Widths = [3.1, 3.1, 2.3, 2.3, 1.9, 1.9, 1.9, 1.7, 1.7, 1.7, 1.7, 1.3, 2.3]
#Line_Styles = ["--", "-.", "--", "-.", "--", "--", "--", "--", "--", "--", "--", "--", "--"]


Plot_Indices = list(range(5,Y.shape[1]))
#Plot_Indices = list(range(23, 99))
Method_Indices = [3,36,38] #
Line_Widths = [2,2,2]
Line_Styles = ["--","-.","-"]
Colors = ["orange","lime","red"]

F = open(os.getcwd() + "\\Prior_Extraction_Info.txt", "rb")
Y = pickle.load(F)[0][Symbol]["Close"]
Y = np.expand_dims(Y, axis = 0)
F.close()

plt.figure(figsize = (17, 8))
plt.rcParams.update({"font.size": 19, "font.family": "Times New Roman"})
plt.title(Symbol)
plt.plot(Plot_Indices, Y[0, Plot_Indices], label = "Input", linewidth = 5.3)

## Critical Points As Boxes
Critical_Point = [23 * i for i in range(1, int(len(Y[0]) / 23))]
if Symbol != "Synthesis":
  F = open(os.getcwd() + "\\..\\..\\Prior_Extraction_Info_2.txt", "rb")
  [Data, Data_Smooth, Data_Critical_Points, Data_Critical_Points_Type] = pickle.load(F)
  F.close()
  if Symbol not in list(Data.keys()):
    F = open(os.getcwd() + "\\..\\..\\Prior_Extraction_Info.txt", "rb")
    [Data, Data_Smooth, Data_Critical_Points, Data_Critical_Points_Type] = pickle.load(F)
    F.close()
  Critical_Point = Data_Critical_Points[Symbol]

Height_Distance = Height_Coefficient * (np.max(Y) - np.min(Y))
Width_Distance = np.ceil(Width_Coefficient * (np.max(Plot_Indices) - np.min(Plot_Indices)))
for i in range(len(Critical_Point)):
  if Critical_Point[i] in Plot_Indices and (Critical_Point[i] - Width_Distance) in Plot_Indices and (Critical_Point[i] + Width_Distance) in Plot_Indices:
    plt.plot([Critical_Point[i] - Width_Distance, Critical_Point[i] + Width_Distance , Critical_Point[i] + Width_Distance, Critical_Point[i] - Width_Distance, Critical_Point[i] - Width_Distance]
                  , [Y[0, Critical_Point[i]] + Height_Distance, Y[0, Critical_Point[i]] + Height_Distance, Y[0, Critical_Point[i]] - Height_Distance, Y[0, Critical_Point[i]] - Height_Distance, Y[0, Critical_Point[i]] + Height_Distance ]
                  , linewidth = 2.7
                  , linestyle = "-"
                  , color = "orangered")

####################### Custom Box #######################

Xs = [78, 95]
Ys = [39, 44]

Xs.sort()
Ys.sort()

plt.plot([Xs[0], Xs[1], Xs[1], Xs[0], Xs[0]],[Ys[0], Ys[0], Ys[1], Ys[1], Ys[0]]
         , linewidth = 2.3
         , linestyle = "-"
         , color = "cyan")

####################### ####################### #######################

#input("SS")
for i in range(len(Method_Indices)):
  plt.plot(Plot_Indices, Y_Plot[Method_Indices[i], Plot_Indices], label = Label_Plot[Method_Indices[i]], linestyle = Line_Styles[i], linewidth = Line_Widths[i], color = Colors[i])
  """
  if "SGD" in Label_Plot[Method_Indices[i]]:
    plt.plot(Plot_Indices, Y_Plot[Method_Indices[i], Plot_Indices], label = Label_Plot[Method_Indices[i]], linestyle = "--", linewidth = 2.3)
  else:
    plt.plot(Plot_Indices, Y_Plot[Method_Indices[i], Plot_Indices], label = Label_Plot[Method_Indices[i]], linestyle = "--")
  """

plt.xlabel("Time value (day) of synthesis stock",fontsize = 30)
plt.ylabel("Closing price of synthesis stock",fontsize = 30)
plt.legend(fontsize = 17, loc = "upper right", framealpha = 0.9)
plt.tight_layout()
plt.savefig(os.getcwd() + "\\"+ Symbol + "_Boxes_Plot_" + str(int(time.time())) + ".png")
#plt.savefig(os.getcwd() + "\\Results\\1672208111\\"+ Symbol + "_Plot_.png")
#plt.savefig(os.getcwd() + "\\Results\\1672208111\\Zoom_"+ Symbol + "_Plot.png")
plt.show()

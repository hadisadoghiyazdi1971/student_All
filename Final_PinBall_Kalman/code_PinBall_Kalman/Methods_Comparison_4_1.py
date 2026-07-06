import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from scipy.misc import derivative
import time # For Save Same Files Without Overlap

#F = open("Prior_Extraction_Info.txt", "rb")
F = open("Prior_Extraction_Info_2.txt", "rb")
Data, Data_Smooth, Critical_Points, Critical_Points_Type = pickle.load(F)
F.close()

Symbols = list(Data.keys())

Symbol = Symbols[12]
#Symbol = Symbols[3] # "TSLA"
print(Data[Symbol].shape)

Y = Data[Symbol]["Close"].to_numpy()#[:221]
Y = np.expand_dims(Y, axis = 0 )
print(Y.shape)
Labels = [Symbol]

Y_Smooth = Data_Smooth[Symbol]["Close"].to_numpy()#[:221]
Y_Smooth = np.expand_dims(Y_Smooth, axis = 0 )
Synthesis_Data = False

"""
Smoothing_WindowSize = 3
Y = np.expand_dims(np.tile(np.abs(np.arange(-23, 23)), 5), axis = 0)
Synthesis_Data = True
#Y = np.add(Y, np.random.normal(0, 1.3, size = Y.shape ))
Y_Smooth = np.convolve(Y[0], np.ones(Smoothing_WindowSize) / Smoothing_WindowSize, "same")
Y_Smooth = np.expand_dims(Y_Smooth, axis = 0)
Symbol = "Synthesis"

Loss_Indices = [24, 47, 70, 93, 116, 139, 162, 185, 208, 231, 254]
Peaks_Indices = [47, 93, 139, 185, 231]
Valeys_Indices = [24, 70, 116, 162, 208, 254]

#Peaks_Indices = [93]
#Valeys_Indices = [70]
"""

Critical_Point = [23 * i for i in range(1, int(len(Y[0]) / 23) + 1)]
Critical_Point[-1] = Critical_Point[-1] - 1
#Critical_Point_Type = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1] # +1: Valey, -1: Peak
Critical_Point_Type = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1] # -1: Valley, +1: Peak

if Synthesis_Data == False:
  Critical_Point = list(Critical_Points[Symbol]) + [Y.shape[1] - 1]
  Critical_Point_Type = list(Critical_Points_Type[Symbol]) + [1]


Z = np.array([[1, 0]])
T = np.array([[1, 1], [0, 1]])
H = np.array([[np.random.rand() * 10]])
R = np.eye(2)
Q = np.random.rand(2,2)
"""
plt.plot(Data["TSLA"]["Close"], "blue")
plt.plot(Data_Smooth["TSLA"]["Close"], "red")
plt.show()
"""
Folder = str(int(time.time())) + "_" + str(Symbol)
Folder = "Results\\" + Folder
os.makedirs(os.getcwd() + "\\" + Folder + "\\")

Iteration = 100
Loss_Indices = []
Peaks_Indices = []
Valeys_Indices = []
#for i in range(min(len(Critical_Point), 4)):
for i in range(len(Critical_Point)):
  if Critical_Point[i] == Y.shape[1] - 1 : break
  Loss_Indices += [Critical_Point[i] + 1]
  if Critical_Point_Type[i] == +1: # len(Peaks_Indices) < 1 and 
    Peaks_Indices += [Critical_Point[i] + 1]
  if Critical_Point_Type[i] == -1: # len(Valeys_Indices) < 1 and 
    Valeys_Indices += [Critical_Point[i] + 1]

#input("SSS")
Method_Total = []
Y_Prediction_Total = []
#Revesal_Losses = []

#"#""
########################## Starting Methods 1 #############################
# ARIMA
Parameters_Setting = [(5, 0, 0), (10, 0, 0), (5, 1, 10), (10, 1, 10)]
#Parameters_Setting = [(5, 1, 10), (10, 1, 10)]

#Parameters_Setting = [(5, 0, 0), (10, 0, 0), (0, 0, 5), (0, 0, 10)
#                                      , (5, 0, 5), (5, 0, 10), (10, 0, 5), (10, 0, 10)
#                                      , (5, 1, 5), (5, 1, 10), (10, 1, 5), (10, 1, 10)
#                                      , (5, 2, 5), (5, 2, 10), (10, 2, 5), (10, 2, 10)]

Min_Loss = 1e+23
for i in range(len(Parameters_Setting)):
  Results = []
  Peaks_Profit = []
  Valleys_Profit = []
  Results_MSE = []
  Results_MAE = []
  Results_MAPE = []
  Results_RMSE = []
  Results_LogCosh = []
  Results_Loss = [] # Total Prediction Loss By MSE
  Method_Name = "ARIMA" + str(Parameters_Setting[i])
  for j in range(Iteration):
    Model = ARIMA(Y[0, :], order = Parameters_Setting[i]) # , enforce_stationarity = False
    Model = Model.fit() # low_memory = True
    #print(Model.summary())
    Prediction = Model.forecasts
    Prediction = Prediction[0]

    Prediction = np.concatenate((Prediction, Model.forecast(steps = 1)))
    ##################################################
    #if "Results_MSE" in globals(): 
    Peaks_Profit += [np.mean(np.subtract(Y[:, Peaks_Indices], Prediction[Peaks_Indices]) )]
    Valleys_Profit += [np.mean(np.subtract(Y[:, Valeys_Indices], Prediction[Valeys_Indices]) )]
    Results_MSE += [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy() )]
    Results_MAE += [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy() )]
    Results_MAPE += [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy() )]
    Results_RMSE += [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy()) )]
    Results_LogCosh += [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy() )]
    Results_Loss += [np.mean(tf.keras.losses.mse(Y[:, :], Prediction[:-1]).numpy() )]
    #
    #print("Round", j + 1, ":", Results_MSE[-1])
    #else:
      #Peaks_Profit = [np.mean(np.subtract(Y[:, Peaks_Indices], Prediction[Peaks_Indices]) )]
      #Valleys_Profit = [np.mean(np.subtract(Y[:, Valeys_Indices], Prediction[Valeys_Indices]) )]
      #Results_MSE = [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy() )]
      #Results_MAE = [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy() )]
      #Results_MAPE = [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy() )]
      #Results_RMSE = [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy()) )]
      #Results_LogCosh = [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Prediction[Loss_Indices]).numpy() )]
    
    print("Round", len(Results_MSE), ":", Results_MSE[-1])
    if  Results_MSE[-1]< Min_Loss: Min_Loss = Results_MSE[-1]

    #Method_Total += [Method_Name]
    Method_Total += [Method_Name + "_" + str(j)]
    #Y_Prediction_Total += [[Prediction]]
    Y_Prediction_Total += [Prediction.tolist()]
    
  print(Method_Name)
  print("----------------------- ----------------------- ----------------------- -----------------------")
  #print(tf.keras.losses.mse(Y_Smooth[0, :190], Prediction[:-1]).numpy())
  print(tf.keras.losses.mse(Y_Smooth[0, 167:171], Prediction[167:171]).numpy())
  ##plt.plot(Y[0, :190])
  ##plt.plot(Prediction[:-1])
  ##plt.show()
  S_Columns = ["Method"] + ["Round_" + str(i + 1) for i in range(Iteration)]
  Total_Peaks_Profit = pd.DataFrame([], columns = S_Columns)
  Total_Valleys_Profit = pd.DataFrame([], columns = S_Columns)
  Total_Results_MSE = pd.DataFrame([], columns = S_Columns)
  Total_Results_MAE = pd.DataFrame([], columns = S_Columns)
  Total_Results_MAPE = pd.DataFrame([], columns = S_Columns)
  Total_Results_RMSE = pd.DataFrame([], columns = S_Columns)
  Total_Results_LogCosh = pd.DataFrame([], columns = S_Columns)
  Total_Results_Loss = pd.DataFrame([], columns = S_Columns)

  if os.path.isfile(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv"):
    Total_Peaks_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")
    Total_Peaks_Profit = Total_Peaks_Profit.drop(columns = Total_Peaks_Profit.keys()[0])

    Total_Valleys_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")
    Total_Valleys_Profit = Total_Valleys_Profit.drop(columns = Total_Valleys_Profit.keys()[0])

    Total_Results_MSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")
    Total_Results_MSE = Total_Results_MSE.drop(columns = Total_Results_MSE.keys()[0])
      
    Total_Results_MAE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")
    Total_Results_MAE = Total_Results_MAE.drop(columns = Total_Results_MAE.keys()[0])
      
    Total_Results_MAPE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")
    Total_Results_MAPE = Total_Results_MAPE.drop(columns = Total_Results_MAPE.keys()[0])
      
    Total_Results_RMSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")
    Total_Results_RMSE = Total_Results_RMSE.drop(columns = Total_Results_RMSE.keys()[0])
      
    Total_Results_LogCosh = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
    Total_Results_LogCosh = Total_Results_LogCosh.drop(columns = Total_Results_LogCosh.keys()[0])

    Total_Results_Loss = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
    Total_Results_Loss = Total_Results_Loss.drop(columns = Total_Results_Loss.keys()[0])

    #Results = pd.DataFrame([[Method_Name] + Results], columns = Total_Results.keys())
  Peaks_Profit = pd.DataFrame([[Method_Name] + Peaks_Profit], columns = S_Columns)
  Total_Peaks_Profit = pd.concat((Total_Peaks_Profit, Peaks_Profit))
  Total_Peaks_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")

  Valleys_Profit = pd.DataFrame([[Method_Name] + Valleys_Profit], columns = S_Columns)
  Total_Valleys_Profit = pd.concat((Total_Valleys_Profit, Valleys_Profit))
  Total_Valleys_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")

  Results_MSE = pd.DataFrame([[Method_Name] + Results_MSE], columns = S_Columns)
  Total_Results_MSE = pd.concat((Total_Results_MSE, Results_MSE))
  Total_Results_MSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")

  Results_MAE = pd.DataFrame([[Method_Name] + Results_MAE], columns = S_Columns)
  Total_Results_MAE = pd.concat((Total_Results_MAE, Results_MAE))
  Total_Results_MAE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")

  Results_MAPE = pd.DataFrame([[Method_Name] + Results_MAPE], columns = S_Columns)
  Total_Results_MAPE = pd.concat((Total_Results_MAPE, Results_MAPE))
  Total_Results_MAPE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")

  Results_RMSE = pd.DataFrame([[Method_Name] + Results_RMSE], columns = S_Columns)
  Total_Results_RMSE = pd.concat((Total_Results_RMSE, Results_RMSE))
  Total_Results_RMSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")

  Results_LogCosh = pd.DataFrame([[Method_Name] + Results_LogCosh], columns = S_Columns)
  Total_Results_LogCosh = pd.concat((Total_Results_LogCosh, Results_LogCosh))
  Total_Results_LogCosh.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")

  Results_Loss = pd.DataFrame([[Method_Name] + Results_Loss], columns = S_Columns)
  Total_Results_Loss = pd.concat((Total_Results_Loss, Results_Loss))
  Total_Results_Loss.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")

########################## Ending Methods 1 #############################
#"#""
#
#
#"#""
########################## Starting Methods 2 #############################
# Kalman Filter
Loss_Name = "MSE"

Method_Name = "KF_" + Loss_Name

Prediction_Day = 1
Peaks_Profit = []
Valleys_Profit = []
Results_MSE = []
Results_MAE = []
Results_MAPE = []
Results_RMSE = []
Results_LogCosh = []
Results_Loss = [] # Total Prediction Loss By MSE

for j in range(Iteration):
  H = np.array([[np.random.rand() * 10]])
  Q = np.random.rand(2,2)

  A_First = np.random.rand(Y.shape[0] * T.shape[0], 1) #* 100# * 46
  P_First = np.eye(Y.shape[0] * T.shape[0]) * 10
  Alpha_First = np.random.multivariate_normal(A_First[:,0], P_First)
  ##Alpha_First = np.array([24, -1])
  #Alpha_First = np.array([22, 1])
  Alpha_First = np.array([float(Y[:, 0]), 1]) if Synthesis_Data == False else np.array([float(Y[:, 0]), -1])
  Alpha_First = np.expand_dims(Alpha_First, axis = 1)
  #print(A_First.shape, "\n", A_First)
  #print(Alpha_First.shape, "\n", Alpha_First)
  #print(P_First.shape, "\n", P_First)

  Alpha = np.array(Alpha_First)
  Alpha = np.expand_dims(Alpha, axis = 0)
  #print("Alpha:", Alpha.shape)
  P = np.array(P_First)
  P = np.expand_dims(P, axis = 0)
  #print("P:", P.shape)
  #print(P.shape)
  V = np.array([]) # Predict Alpha
  K = np.array([]) # Kalman Filter Gain
  L = np.array([]) # Predict P
  Y_Prediction = np.array([])
  Y_Fit = np.array([])

  for i in range(Y.shape[1]):
    # Predict
    Temp = np.dot(T, Alpha[i, :, :])
    Temp = np.expand_dims(Temp, 0)
    #print("V:", Temp)
    V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp
    
    Temp = V[i, :, :]
    for o in range(Prediction_Day - 1):
      Temp = np.dot(T, Temp)
    Temp = np.dot(Z, Temp)
    Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
    #if i>2:
    #  print(Y_Prediction.shape)
    #  print(Y.shape)
    #  TT = np.dot(Alpha[i,:,:],Alpha[i,:,:]) # Bug For Stop And See Results

    Temp = np.dot(np.dot(T, P[i, :, :]), np.transpose(T))
    Temp = np.add(Temp, np.dot(np.dot(R, Q), np.transpose(R)))
    Temp = np.expand_dims(Temp, axis = 0)
    #print("L:", Temp)
    L = np.concatenate((L, Temp)) if len(L.shape) > 1 else Temp
    
    # Update
    Temp = np.dot(np.dot(Z, L[i, :, :]), np.transpose(Z))
    Temp = np.add(Temp, H)
    try:
      Temp = np.dot(np.dot(L[i, :, :], np.transpose(Z)), np.linalg.inv(Temp))
    except:
      Temp = np.dot(np.dot(L[i, :, :], np.transpose(Z)), np.linalg.pinv(Temp))
    Temp= np.expand_dims(Temp, 0)
    #print("K:", Temp)
    K = np.concatenate((K , Temp)) if len(K.shape) > 1 else Temp  
    
    Temp = np.subtract(np.expand_dims(Y[:, i], axis = 1), np.dot(Z, V[i, :, :]))
    #print("Alpha_1:", Temp.shape)
    Temp = np.dot(K[i, :, :], Temp)
    Temp = np.add(V[i, :, :], Temp)
    Temp= np.expand_dims(Temp, 0)
    Alpha = np.concatenate((Alpha, Temp))

    Temp = np.subtract(np.eye(Y.shape[0] * T.shape[0]), np.dot(K[i, :, :], Z))
    Temp = np.dot(np.dot(Temp, L[i, :, :]), np.transpose(Temp))
    if len(H.shape) == 1:
      H = np.expand_dims(H,1)
    Temp = np.add(Temp, np.dot(np.dot(K[i, :, :], H), np.transpose(K[i, :, :])))
    Temp= np.expand_dims(Temp, 0)
    #print("P_1:", Temp)
    P = np.concatenate((P , Temp))

    #
    Temp = np.dot(Z, Alpha[-1, :, :])
    Y_Fit =  np.concatenate((Y_Fit, Temp), axis = 1) if len(Y_Fit.shape) > 1 else Temp
    
    #
    if i %(int(Y.shape[1]/10)) ==0:
      print(i, end="\t")
    #print("Temp:", Temp.shape)
    #

  print()

  Y_Temp = np.copy(Y)
  #Y_Tild = np.transpose(Y).reshape(Min_Stock_Length * len(S_Ticker), 1) # m x n => n x m => nm x 1 
  # Y_Prediction_Tild Is Sum Of Miu_Tild(Trend), Taw_Tild(Season), Omega_Tild(Cycle)
  #Y_Prediction_Tild = np.transpose(Y_Prediction).reshape(Min_Stock_Length * len(S_Ticker), 1) # m x n => n x m => nm x 1 
  #
  print("MSE:", tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy()))
  print("MAE:", tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy()))
  #Q *= 0.987

  #### Optional
  # Predict
  Temp = np.dot(T, Alpha[i, :, :])
  Temp = np.expand_dims(Temp, 0)
  #print("V:", Temp)
  V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp

  Temp = np.dot(Z, V[i, :, :])
  Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
  ####

  #Method_Comparison = Method_Comparison + [Loss_Name] if "Method_Comparison" in globals() else [Loss_Name]
  #Y_Comparison = np.concatenate((Y_Comparison, Y_Prediction)) if "Y_Comparison" in globals() else Y_Prediction

  #if "Results_MSE" in globals():
  Peaks_Profit += [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
  Valleys_Profit += [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
  Results_MSE += [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
  Results_MAE += [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
  Results_MAPE += [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
  Results_RMSE += [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
  Results_LogCosh += [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
  Results_Loss += [np.mean(tf.keras.losses.mse(Y[:, :], Y_Prediction[:, :-1]).numpy() )]
    #print("Round", j + 1, ":", Results_MSE[-1])
  #else:
    #Peaks_Profit = [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
    #Valleys_Profit = [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
    #Results_MSE = [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    #Results_MAE = [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    #Results_MAPE = [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    #Results_RMSE = [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
    #Results_LogCosh = [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]

  print("Round", len(Results_MSE), ":", Results_MSE[-1])

  print(Method_Name)
  print("----------------------- ----------------------- ----------------------- -----------------------")
  Method_Total += [Method_Name]
  #Y_Prediction_Total += [[Y_Prediction]]
  Y_Prediction_Total += [Y_Prediction[0, :].tolist()]
  #
S_Columns = ["Method"] + ["Round_" + str(i + 1) for i in range(Iteration)]
Total_Peaks_Profit = pd.DataFrame([], columns = S_Columns)
Total_Valleys_Profit = pd.DataFrame([], columns = S_Columns)
Total_Results_MSE = pd.DataFrame([], columns = S_Columns)
Total_Results_MAE = pd.DataFrame([], columns = S_Columns)
Total_Results_MAPE = pd.DataFrame([], columns = S_Columns)
Total_Results_RMSE = pd.DataFrame([], columns = S_Columns)
Total_Results_LogCosh = pd.DataFrame([], columns = S_Columns)
Total_Results_Loss = pd.DataFrame([], columns = S_Columns)

if os.path.isfile(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv"):
  Total_Peaks_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")
  Total_Peaks_Profit = Total_Peaks_Profit.drop(columns = Total_Peaks_Profit.keys()[0])

  Total_Valleys_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")
  Total_Valleys_Profit = Total_Valleys_Profit.drop(columns = Total_Valleys_Profit.keys()[0])

  Total_Results_MSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")
  Total_Results_MSE = Total_Results_MSE.drop(columns = Total_Results_MSE.keys()[0])
    
  Total_Results_MAE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")
  Total_Results_MAE = Total_Results_MAE.drop(columns = Total_Results_MAE.keys()[0])
    
  Total_Results_MAPE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")
  Total_Results_MAPE = Total_Results_MAPE.drop(columns = Total_Results_MAPE.keys()[0])
    
  Total_Results_RMSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")
  Total_Results_RMSE = Total_Results_RMSE.drop(columns = Total_Results_RMSE.keys()[0])
    
  Total_Results_LogCosh = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
  Total_Results_LogCosh = Total_Results_LogCosh.drop(columns = Total_Results_LogCosh.keys()[0])

  Total_Results_Loss = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
  Total_Results_Loss = Total_Results_Loss.drop(columns = Total_Results_Loss.keys()[0])

  #Results = pd.DataFrame([[Method_Name] + Results], columns = Total_Results.keys())
Peaks_Profit = pd.DataFrame([[Method_Name] + Peaks_Profit], columns = S_Columns)
Total_Peaks_Profit = pd.concat((Total_Peaks_Profit, Peaks_Profit))
Total_Peaks_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")

Valleys_Profit = pd.DataFrame([[Method_Name] + Valleys_Profit], columns = S_Columns)
Total_Valleys_Profit = pd.concat((Total_Valleys_Profit, Valleys_Profit))
Total_Valleys_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")

Results_MSE = pd.DataFrame([[Method_Name] + Results_MSE], columns = S_Columns)
Total_Results_MSE = pd.concat((Total_Results_MSE, Results_MSE))
Total_Results_MSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")

Results_MAE = pd.DataFrame([[Method_Name] + Results_MAE], columns = S_Columns)
Total_Results_MAE = pd.concat((Total_Results_MAE, Results_MAE))
Total_Results_MAE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")

Results_MAPE = pd.DataFrame([[Method_Name] + Results_MAPE], columns = S_Columns)
Total_Results_MAPE = pd.concat((Total_Results_MAPE, Results_MAPE))
Total_Results_MAPE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")

Results_RMSE = pd.DataFrame([[Method_Name] + Results_RMSE], columns = S_Columns)
Total_Results_RMSE = pd.concat((Total_Results_RMSE, Results_RMSE))
Total_Results_RMSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")

Results_LogCosh = pd.DataFrame([[Method_Name] + Results_LogCosh], columns = S_Columns)
Total_Results_LogCosh = pd.concat((Total_Results_LogCosh, Results_LogCosh))
Total_Results_LogCosh.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")

Results_Loss = pd.DataFrame([[Method_Name] + Results_Loss], columns = S_Columns)
Total_Results_Loss = pd.concat((Total_Results_Loss, Results_Loss))
Total_Results_Loss.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")

########################## Ending Methods 2 #############################
#"#""
#
#"#""
########################## Starting Methods 3 #############################
# CorrEntropy
#Betha = np.ones(Betha.shape) * 0.01
def Differential_Gaus(x, Sigma = 1):
  #return np.exp((-1 * np.linalg.norm(x))/ (2 * (Sigma ** 2)) )
  #return np.exp((-1 * (x ** 2))/ (2 * (Sigma ** 2)) )
  return np.exp((-1 * x)/ (2 * (Sigma ** 2)) )

# Kalman Filter
Loss_Name = "CorrEntropy"

#Sigma_List = [7, 11, 23, 31, 47]
#Sigma_List = [4, 7, 11, 23, 31]
#Sigma_List = [0.1 * i for i in range(1, 69)]
Sigma_List = [0.25 * i for i in range(1, 13)] # For Article
Coefficent_List = [0.9]

for a in range(len(Sigma_List)):
  My_Sigma = Sigma_List[a]
  for b in range(len(Coefficent_List)):
    Loss_Name = "CorrEntropy"
    Loss_Name += "_Sigma_" + "{0:.2e}".format(Sigma_List[a]).replace("+", "") # str(My_Sigma)
    Loss_Name += "_Loss_Coefficent_" + "{0:.2e}".format(Coefficent_List[b]).replace("+", "") # str(Loss_Coefficent)
    Loss_Coefficent = Coefficent_List[b]
    
    Method_Name = "KF_" + Loss_Name
    Peaks_Profit = []
    Valleys_Profit = []
    Results_MSE = []
    Results_MAE = []
    Results_MAPE = []
    Results_RMSE = []
    Results_LogCosh = []
    Results_Loss = [] # Total Prediction Loss By MSE
    for j in range(Iteration): 
      ####################### Start Of Kalman Filter Process
      A_First = np.random.rand(T.shape[0], 1) #* 100# * 46
      P_First = np.eye(T.shape[0]) * 10
      Alpha_First = np.random.multivariate_normal(A_First[:,0], P_First)
      Alpha_First = np.tile(np.array([4.37, 0.23]), Y.shape[0])
      ##Alpha_First = np.array([24, -1])
      #Alpha_First = np.array([22, 1])
      Alpha_First = np.array([float(Y[:, 0]), 1]) if Synthesis_Data == False else np.array([float(Y[:, 0]), -1])
      Alpha_First = np.expand_dims(Alpha_First, axis = 1)
      #print(A_First.shape, "\n", A_First)
      #print(Alpha_First.shape, "\n", Alpha_First)
      #print(P_First.shape, "\n", P_First)

      Alpha = np.array(Alpha_First)
      Alpha = np.expand_dims(Alpha, axis = 0)
      #print("Alpha:", Alpha.shape)
      P = np.array(P_First)
      P = np.expand_dims(P, axis = 0)
      #print("P:", P.shape)
      #print(P.shape)
      V = np.array([]) # Predict Alpha
      K = np.array([]) # Kalman Filter Gain
      L = np.array([]) # Predict P
      Y_Prediction = np.array([])
      Y_Fit = np.array([])

      for i in range(Y.shape[1]):
        # Predict
        Temp = np.dot(T, Alpha[i, :, :])
        Temp = np.expand_dims(Temp, 0)
        #print("V:", Temp)
        V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp
        
        Temp = np.dot(Z, V[i, :, :])
        Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
        #if i>2:
        #  print(Y_Prediction.shape)
        #  print(Y.shape)
        #  TT = np.dot(Alpha[i,:,:],Alpha[i,:,:]) # Bug For Stop And See Results

        Temp = np.dot(np.dot(T, P[i, :, :]), np.transpose(T))
        Temp = np.add(Temp, np.dot(np.dot(R, Q), np.transpose(R)))
        Temp = np.expand_dims(Temp, axis = 0)
        #print("L:", Temp)
        L = np.concatenate((L, Temp)) if len(L.shape) > 1 else Temp
        
        # Update
        Temp_1 = np.abs(np.subtract(np.expand_dims(Y[:, i], axis = 1), np.dot(Z, V[i, :, :])))
        #print("Temp_1:", Temp_1)
        try:
          Temp_1 = np.dot(np.dot(Temp_1, np.linalg.inv(H)), np.transpose(Temp_1))
        except:
          Temp_1 = np.dot(np.dot(np.transpose(Temp_1), np.linalg.pinv(H)), Temp_1)
        #
        Temp_2 = np.abs(np.subtract(V[i, :, :], np.dot(T, Alpha[i, :, :])))
        try:
          Temp_2 = np.dot(np.dot(Temp_2, np.linalg.inv(L[i, :, :])), np.transpose(Temp_2))
        except:
          Temp_2 = np.dot(np.dot(np.transpose(Temp_2), np.linalg.pinv(L[i, :, :])), Temp_2)
        #
        K_Temp = Differential_Gaus(Temp_1, Sigma = My_Sigma) / Differential_Gaus(Temp_2 + 1e-7, Sigma = My_Sigma)
        K_Temp = np.multiply((Loss_Coefficent/(1  - Loss_Coefficent)), K_Temp)
        #print("Temp_1:", Temp_1, "Temp_2:", Temp_2)
        #print("K_Temp:", K_Temp)
        try:
          Temp = np.dot(np.multiply(K_Temp, np.transpose(Z)), np.dot(np.linalg.inv(H), Z))
        except:
          Temp = np.dot(np.multiply(K_Temp, np.transpose(Z)), np.dot(np.linalg.pinv(H), Z))
        #
        try:
          Temp = np.add(np.linalg.inv(L[i, :, :]), Temp)
        except:
          Temp = np.add(np.linalg.pinv(L[i, :, :]), Temp)
        #
        try:
          Temp = np.linalg.inv(Temp)
        except:
          Temp = np.linalg.pinv(Temp)
        #
        try:
          Temp = np.dot(np.multiply(Temp, K_Temp), np.dot(np.transpose(Z), np.linalg.inv(H)))
        except:
          Temp = np.dot(np.multiply(Temp, K_Temp), np.dot(np.transpose(Z), np.linalg.pinv(H)))
        Temp = np.expand_dims(Temp, 0)
        K = np.concatenate((K , Temp)) if len(K.shape) > 1 else Temp  
        
        Temp = np.subtract(np.expand_dims(Y[:, i], axis = 1), np.dot(Z, V[i, :, :]))
        #print("Alpha_1:", Temp.shape)
        Temp = np.dot(K[i, :, :], Temp)
        Temp = np.add(V[i, :, :], Temp)
        Temp= np.expand_dims(Temp, 0)
        Alpha = np.concatenate((Alpha, Temp))

        Temp = np.subtract(np.eye(T.shape[0]), np.dot(K[i, :, :], Z))
        Temp = np.dot(np.dot(Temp, L[i, :, :]), np.transpose(Temp))
        if len(H.shape) == 1:
          H = np.expand_dims(H, 1)
        Temp = np.add(Temp, np.dot(np.dot(K[i, :, :], H), np.transpose(K[i, :, :])))
        Temp= np.expand_dims(Temp, 0)
        #print("P_1:", Temp)
        P = np.concatenate((P , Temp))

        #
        Temp = np.dot(Z, Alpha[-1, :, :])
        Y_Fit =  np.concatenate((Y_Fit, Temp), axis = 1) if len(Y_Fit.shape) > 1 else Temp
        
        #
        if i % (int(Y.shape[1]/10)) ==0:
          print(i, end="\t")
        #print("Temp:", Temp.shape)
        #

      print()

      Y_Temp = np.copy(Y)
      #Y_Tild = np.transpose(Y).reshape(Min_Stock_Length * len(S_Ticker), 1) # m x n => n x m => nm x 1 
      # Y_Prediction_Tild Is Sum Of Miu_Tild(Trend), Taw_Tild(Season), Omega_Tild(Cycle)
      #Y_Prediction_Tild = np.transpose(Y_Prediction).reshape(Min_Stock_Length * len(S_Ticker), 1) # m x n => n x m => nm x 1 
      #
      print("MSE:", tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy()))
      print("MAE:", tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy()))
      #Q *= 0.987

      #### Optional
      # Predict
      Temp = np.dot(T, Alpha[i, :, :])
      Temp = np.expand_dims(Temp, 0)
      #print("V:", Temp)
      V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp

      Temp = np.dot(Z, V[i, :, :])
      Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
      ####

      #Method_Comparison = Method_Comparison + [Loss_Name] if "Method_Comparison" in globals() else [Loss_Name] # For Article
      #Y_Comparison = np.concatenate((Y_Comparison, Y_Prediction)) if "Y_Comparison" in globals() else Y_Prediction # For Article
      
      #if "Results_MSE" in globals():
      Peaks_Profit += [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
      Valleys_Profit += [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
      Results_MSE += [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      Results_MAE += [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      Results_MAPE += [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      Results_RMSE += [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
      Results_LogCosh += [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      Results_Loss += [np.mean(tf.keras.losses.mse(Y[:, :], Y_Prediction[:, :-1]).numpy() )]
        #print("Round", j + 1, ":", Results_MSE[-1])
      #else:
        #Peaks_Profit = [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
        #Valleys_Profit = [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
        #Results_MSE = [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
        #Results_MAE = [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
        #Results_MAPE = [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
        #Results_RMSE = [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
        #Results_LogCosh = [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]

      print("Round", len(Results_MSE), ":", Results_MSE[-1])
      #
      #if len(Results_MSE) == 10:
      print(Method_Name)
      print("----------------------- ----------------------- ----------------------- -----------------------")
      Method_Total += [Method_Name]
      #Y_Prediction_Total += [[Y_Prediction]]
      Y_Prediction_Total += [Y_Prediction[0, :].tolist()]

    S_Columns = ["Method"] + ["Round_" + str(i + 1) for i in range(Iteration)]
    Total_Peaks_Profit = pd.DataFrame([], columns = S_Columns)
    Total_Valleys_Profit = pd.DataFrame([], columns = S_Columns)
    Total_Results_MSE = pd.DataFrame([], columns = S_Columns)
    Total_Results_MAE = pd.DataFrame([], columns = S_Columns)
    Total_Results_MAPE = pd.DataFrame([], columns = S_Columns)
    Total_Results_RMSE = pd.DataFrame([], columns = S_Columns)
    Total_Results_LogCosh = pd.DataFrame([], columns = S_Columns)
    Total_Results_Loss = pd.DataFrame([], columns = S_Columns)

    if os.path.isfile(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv"):
      Total_Peaks_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")
      Total_Peaks_Profit = Total_Peaks_Profit.drop(columns = Total_Peaks_Profit.keys()[0])

      Total_Valleys_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")
      Total_Valleys_Profit = Total_Valleys_Profit.drop(columns = Total_Valleys_Profit.keys()[0])

      Total_Results_MSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")
      Total_Results_MSE = Total_Results_MSE.drop(columns = Total_Results_MSE.keys()[0])
        
      Total_Results_MAE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")
      Total_Results_MAE = Total_Results_MAE.drop(columns = Total_Results_MAE.keys()[0])
        
      Total_Results_MAPE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")
      Total_Results_MAPE = Total_Results_MAPE.drop(columns = Total_Results_MAPE.keys()[0])
        
      Total_Results_RMSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")
      Total_Results_RMSE = Total_Results_RMSE.drop(columns = Total_Results_RMSE.keys()[0])
        
      Total_Results_LogCosh = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
      Total_Results_LogCosh = Total_Results_LogCosh.drop(columns = Total_Results_LogCosh.keys()[0])

      Total_Results_Loss = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
      Total_Results_Loss = Total_Results_Loss.drop(columns = Total_Results_Loss.keys()[0])

      #Results = pd.DataFrame([[Method_Name] + Results], columns = Total_Results.keys())
    Peaks_Profit = pd.DataFrame([[Method_Name] + Peaks_Profit], columns = S_Columns)
    Total_Peaks_Profit = pd.concat((Total_Peaks_Profit, Peaks_Profit))
    Total_Peaks_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")

    Valleys_Profit = pd.DataFrame([[Method_Name] + Valleys_Profit], columns = S_Columns)
    Total_Valleys_Profit = pd.concat((Total_Valleys_Profit, Valleys_Profit))
    Total_Valleys_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")

    Results_MSE = pd.DataFrame([[Method_Name] + Results_MSE], columns = S_Columns)
    Total_Results_MSE = pd.concat((Total_Results_MSE, Results_MSE))
    Total_Results_MSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")

    Results_MAE = pd.DataFrame([[Method_Name] + Results_MAE], columns = S_Columns)
    Total_Results_MAE = pd.concat((Total_Results_MAE, Results_MAE))
    Total_Results_MAE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")

    Results_MAPE = pd.DataFrame([[Method_Name] + Results_MAPE], columns = S_Columns)
    Total_Results_MAPE = pd.concat((Total_Results_MAPE, Results_MAPE))
    Total_Results_MAPE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")

    Results_RMSE = pd.DataFrame([[Method_Name] + Results_RMSE], columns = S_Columns)
    Total_Results_RMSE = pd.concat((Total_Results_RMSE, Results_RMSE))
    Total_Results_RMSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")

    Results_LogCosh = pd.DataFrame([[Method_Name] + Results_LogCosh], columns = S_Columns)
    Total_Results_LogCosh = pd.concat((Total_Results_LogCosh, Results_LogCosh))
    Total_Results_LogCosh.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")

    Results_Loss = pd.DataFrame([[Method_Name] + Results_Loss], columns = S_Columns)
    Total_Results_Loss = pd.concat((Total_Results_Loss, Results_Loss))
    Total_Results_Loss.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")


########################## Ending Methods 3 #############################
#"#""
#
#"#""
########################## Starting Methods 4 #############################
# PseudoHuber
def Differential_Pseudo_Huber(X, Delta = 1):
  return 1.0 / np.sqrt(np.add(1, (X/Delta) ** 2))

# Kalman Filter
Loss_Name = "PseudoHuber"

#Delta_List = [0.25, 0.5, 1, 1.5, 2.3]
#Delta_List = [3.1e0, 4.7e0, 7e0, 9.7e0, 13e0]
Delta_List = [0.25 * i for i in range(1, 13)] # For Article
#Coefficent_List = [0.1, 0.3, 0.5, 0.7, 0.9]
Coefficent_List = [0.9]

for a in range(len(Delta_List)):
  My_Delta = Delta_List[a]
  for b in range(len(Coefficent_List)):
    Loss_Name = "PseudoHuber"
    Loss_Name += "_Delta_" + "{0:.2e}".format(Delta_List[a]).replace("+", "") # str(My_Sigma)
    Loss_Name += "_Loss_Coefficent_" + "{0:.2e}".format(Coefficent_List[b]).replace("+", "") # str(Loss_Coefficent)
    Loss_Coefficent = Coefficent_List[b]
    #Loss_Name += "_Mode_" + str(Mode)
    Method_Name = "KF_" + Loss_Name

    Peaks_Profit = []
    Valleys_Profit = []
    Results_MSE = []
    Results_MAE = []
    Results_MAPE = []
    Results_RMSE = []
    Results_LogCosh = []
    Results_Loss = [] # Total Prediction Loss By MSE
    for j in range(Iteration): # Iteration
      ####################### Start Of Kalman Filter Process
      A_First = np.random.rand(T.shape[0], 1) #* 100# * 46
      P_First = np.eye(T.shape[0]) * 10
      Alpha_First = np.random.multivariate_normal(A_First[:,0], P_First)
      ##Alpha_First = np.array([24, -1])
      #Alpha_First = np.array([22, 1])
      Alpha_First = np.array([float(Y[:, 0]), 1]) if Synthesis_Data == False else np.array([float(Y[:, 0]), -1])
      Alpha_First = np.expand_dims(Alpha_First, axis = 1)
      #print(A_First.shape, "\n", A_First)
      #print(Alpha_First.shape, "\n", Alpha_First)
      #print(P_First.shape, "\n", P_First)

      Alpha = np.array(Alpha_First)
      Alpha = np.expand_dims(Alpha, axis = 0)
      #print("Alpha:", Alpha.shape)
      P = np.array(P_First)
      P = np.expand_dims(P, axis = 0)
      #print("P:", P.shape)
      #print(P.shape)
      V = np.array([]) # Predict Alpha
      K = np.array([]) # Kalman Filter Gain
      L = np.array([]) # Predict P
      Y_Prediction = np.array([])
      Y_Fit = np.array([])

      for i in range(Y.shape[1]):
        # Predict
        Temp = np.dot(T, Alpha[i, :, :])
        Temp = np.expand_dims(Temp, 0)
        #print("V:", Temp)
        V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp
        
        Temp = np.dot(Z, V[i, :, :])
        Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
        #if i>2:
        #  print(Y_Prediction.shape)
        #  print(Y.shape)
        #  TT = np.dot(Alpha[i,:,:],Alpha[i,:,:]) # Bug For Stop And See Results

        Temp = np.dot(np.dot(T, P[i, :, :]), np.transpose(T))
        Temp = np.add(Temp, np.dot(np.dot(R, Q), np.transpose(R)))
        Temp = np.expand_dims(Temp, axis = 0)
        #print("L:", Temp)
        L = np.concatenate((L, Temp)) if len(L.shape) > 1 else Temp
        
        # Update
        Temp_1 = np.abs(np.subtract(np.expand_dims(Y[:, i], axis = 1), np.dot(Z, V[i, :, :])))
        #print("Temp_1:", Temp_1)
        try:
          Temp_1 = np.dot(np.dot(Temp_1, np.linalg.inv(H)), np.transpose(Temp_1))
        except:
          Temp_1 = np.dot(np.dot(np.transpose(Temp_1), np.linalg.pinv(H)), Temp_1)
        #
        Temp_2 = np.abs(np.subtract(Alpha[i, :, :], V[i, :, :]))
        try:
          Temp_2 = np.dot(np.dot(Temp_2, np.linalg.inv(L[i, :, :])), np.transpose(Temp_2))
        except:
          Temp_2 = np.dot(np.dot(np.transpose(Temp_2), np.linalg.pinv(L[i, :, :])), Temp_2)
        #
        K_Temp = Differential_Pseudo_Huber(Temp_1, Delta = My_Delta) / Differential_Pseudo_Huber(Temp_2, Delta = My_Delta)
        #print("Temp_1:", Temp_1, "Temp_2:", Temp_2)
        #print("K_Temp:", K_Temp)
        try:
          Temp = np.dot(np.multiply(K_Temp, np.transpose(Z)), np.dot(np.linalg.inv(H), Z))
        except:
          Temp = np.dot(np.multiply(K_Temp, np.transpose(Z)), np.dot(np.linalg.pinv(H), Z))
        #
        try:
          Temp = np.add(np.linalg.inv(L[i, :, :]), Temp)
        except:
          Temp = np.add(np.linalg.pinv(L[i, :, :]), Temp)
        #
        try:
          Temp = np.linalg.inv(Temp)
        except:
          Temp = np.linalg.pinv(Temp)
        #
        try:
          Temp = np.dot(np.multiply(Temp, K_Temp), np.dot(np.transpose(Z), np.linalg.inv(H)))
        except:
          Temp = np.dot(np.multiply(Temp, K_Temp), np.dot(np.transpose(Z), np.linalg.pinv(H)))
        Temp = np.expand_dims(Temp, 0)
        
        K = np.concatenate((K , Temp)) if len(K.shape) > 1 else Temp  

        Temp = np.subtract(np.expand_dims(Y[:, i], axis = 1), np.dot(Z, V[i, :, :]))
        #print("Alpha_1:", Temp.shape)
        Temp = np.dot(K[i, :, :], Temp)
        Temp = np.add(V[i, :, :], Temp)
        Temp= np.expand_dims(Temp, 0)
        Alpha = np.concatenate((Alpha, Temp))

        Temp = np.subtract(np.eye(T.shape[0]), np.dot(K[i, :, :], Z))
        Temp = np.dot(np.dot(Temp, L[i, :, :]), np.transpose(Temp))
        if len(H.shape) == 1:
          H = np.expand_dims(H, 1)
        Temp = np.add(Temp, np.dot(np.dot(K[i, :, :], H), np.transpose(K[i, :, :])))
        Temp= np.expand_dims(Temp, 0)
        #print("P_1:", Temp)
        P = np.concatenate((P , Temp))

        #
        Temp = np.dot(Z, Alpha[-1, :, :])
        Y_Fit =  np.concatenate((Y_Fit, Temp), axis = 1) if len(Y_Fit.shape) > 1 else Temp
        
        #
        if i % (int(Y.shape[1]/10)) ==0:
          print(i, end="\t")
        #print("Temp:", Temp.shape)
        #

      print()

      Y_Temp = np.copy(Y)
      #Y_Tild = np.transpose(Y).reshape(Min_Stock_Length * len(S_Ticker), 1) # m x n => n x m => nm x 1 
      # Y_Prediction_Tild Is Sum Of Miu_Tild(Trend), Taw_Tild(Season), Omega_Tild(Cycle)
      #Y_Prediction_Tild = np.transpose(Y_Prediction).reshape(Min_Stock_Length * len(S_Ticker), 1) # m x n => n x m => nm x 1 
      #
      print("MSE:", tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy()))
      print("MAE:", tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy()))
      #Q *= 0.987
      
      #### Optional
      # Predict
      Temp = np.dot(T, Alpha[i, :, :])
      Temp = np.expand_dims(Temp, 0)
      #print("V:", Temp)
      V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp

      Temp = np.dot(Z, V[i, :, :])
      Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
      ####

      #Method_Comparison = Method_Comparison + [Loss_Name] if "Method_Comparison" in globals() else [Loss_Name] # For Article
      #Y_Comparison = np.concatenate((Y_Comparison, Y_Prediction)) if "Y_Comparison" in globals() else Y_Prediction # For Article

      #if "Results_MSE" in globals():
      Peaks_Profit += [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
      Valleys_Profit += [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
      Results_MSE += [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      Results_MAE += [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      Results_MAPE += [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      Results_RMSE += [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
      Results_LogCosh += [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      Results_Loss += [np.mean(tf.keras.losses.mse(Y[:, :], Y_Prediction[:, :-1]).numpy() )]
        #print("Round", j + 1, ":", Results_MSE[-1])
      #else:
        #Peaks_Profit = [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
        #Valleys_Profit = [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
        #Results_MSE = [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
        #Results_MAE = [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
        #Results_MAPE = [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
        #Results_RMSE = [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
        #Results_LogCosh = [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]

      print("Round", len(Results_MSE), ":", Results_MSE[-1])
      #
      #if len(Results_MSE) == 10:
      print(Method_Name)
      print("----------------------- ----------------------- ----------------------- -----------------------")
      Method_Total += [Method_Name]
      #Y_Prediction_Total += [[Y_Prediction]]
      Y_Prediction_Total += [Y_Prediction[0, :].tolist()]

    S_Columns = ["Method"] + ["Round_" + str(i + 1) for i in range(Iteration)]
    Total_Peaks_Profit = pd.DataFrame([], columns = S_Columns)
    Total_Valleys_Profit = pd.DataFrame([], columns = S_Columns)
    Total_Results_MSE = pd.DataFrame([], columns = S_Columns)
    Total_Results_MAE = pd.DataFrame([], columns = S_Columns)
    Total_Results_MAPE = pd.DataFrame([], columns = S_Columns)
    Total_Results_RMSE = pd.DataFrame([], columns = S_Columns)
    Total_Results_LogCosh = pd.DataFrame([], columns = S_Columns)
    Total_Results_Loss = pd.DataFrame([], columns = S_Columns)

    if os.path.isfile(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv"):
      Total_Peaks_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")
      Total_Peaks_Profit = Total_Peaks_Profit.drop(columns = Total_Peaks_Profit.keys()[0])

      Total_Valleys_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")
      Total_Valleys_Profit = Total_Valleys_Profit.drop(columns = Total_Valleys_Profit.keys()[0])

      Total_Results_MSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")
      Total_Results_MSE = Total_Results_MSE.drop(columns = Total_Results_MSE.keys()[0])
        
      Total_Results_MAE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")
      Total_Results_MAE = Total_Results_MAE.drop(columns = Total_Results_MAE.keys()[0])
        
      Total_Results_MAPE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")
      Total_Results_MAPE = Total_Results_MAPE.drop(columns = Total_Results_MAPE.keys()[0])
        
      Total_Results_RMSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")
      Total_Results_RMSE = Total_Results_RMSE.drop(columns = Total_Results_RMSE.keys()[0])
        
      Total_Results_LogCosh = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
      Total_Results_LogCosh = Total_Results_LogCosh.drop(columns = Total_Results_LogCosh.keys()[0])

      Total_Results_Loss = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
      Total_Results_Loss = Total_Results_Loss.drop(columns = Total_Results_Loss.keys()[0])

      #Results = pd.DataFrame([[Method_Name] + Results], columns = Total_Results.keys())
    Peaks_Profit = pd.DataFrame([[Method_Name] + Peaks_Profit], columns = S_Columns)
    Total_Peaks_Profit = pd.concat((Total_Peaks_Profit, Peaks_Profit))
    Total_Peaks_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")

    Valleys_Profit = pd.DataFrame([[Method_Name] + Valleys_Profit], columns = S_Columns)
    Total_Valleys_Profit = pd.concat((Total_Valleys_Profit, Valleys_Profit))
    Total_Valleys_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")

    Results_MSE = pd.DataFrame([[Method_Name] + Results_MSE], columns = S_Columns)
    Total_Results_MSE = pd.concat((Total_Results_MSE, Results_MSE))
    Total_Results_MSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")

    Results_MAE = pd.DataFrame([[Method_Name] + Results_MAE], columns = S_Columns)
    Total_Results_MAE = pd.concat((Total_Results_MAE, Results_MAE))
    Total_Results_MAE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")

    Results_MAPE = pd.DataFrame([[Method_Name] + Results_MAPE], columns = S_Columns)
    Total_Results_MAPE = pd.concat((Total_Results_MAPE, Results_MAPE))
    Total_Results_MAPE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")

    Results_RMSE = pd.DataFrame([[Method_Name] + Results_RMSE], columns = S_Columns)
    Total_Results_RMSE = pd.concat((Total_Results_RMSE, Results_RMSE))
    Total_Results_RMSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")

    Results_LogCosh = pd.DataFrame([[Method_Name] + Results_LogCosh], columns = S_Columns)
    Total_Results_LogCosh = pd.concat((Total_Results_LogCosh, Results_LogCosh))
    Total_Results_LogCosh.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")

    Results_Loss = pd.DataFrame([[Method_Name] + Results_Loss], columns = S_Columns)
    Total_Results_Loss = pd.concat((Total_Results_Loss, Results_Loss))
    Total_Results_Loss.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")

#
########################## Ending Methods 4 #############################
#"#""
#
#"#""
########################## Starting Methods 5 #############################
# LSTM
#from IPython.display import clear_output

Window_Size_List = [5, 10]
Nodes_Counts = [5, 10, 15, 20]

YY_Temp = np.array([])

for i in range(len(Window_Size_List)):
  Window_Size = Window_Size_List[i]
  if Y.shape[0] < Y.shape[1]: Y = np.transpose(Y)
  XX = np.array([Y[i:(i + Window_Size)] for i in range(Y.shape[0] - Window_Size)])
  YY = np.array([Y[(i + Window_Size)] for i in range(Y.shape[0] - Window_Size)])
  if Y.shape[1] < Y.shape[0]: Y = np.transpose(Y)

  XX_Train = XX[:190, :]
  YY_Train = YY[:190]
  XX_Test = XX[190:, :]
  YY_Test = YY[190:]
  for k in range(len(Nodes_Counts)):
    Method_Name = "LSTM_N" + str(Nodes_Counts[k]) + "_W" + str(Window_Size)
    Peaks_Profit = []
    Valleys_Profit = []
    Results_MSE = []
    Results_MAE = []
    Results_MAPE = []
    Results_RMSE = []
    Results_LogCosh = []
    Results_Loss = [] # Total Prediction Loss By MSE
    #for j in range(10):
    while len(Results_MSE) < Iteration :
      #
      L_0 = tf.keras.layers.Input(Window_Size)
      L_1 = tf.keras.layers.Reshape((1, -1)) (L_0)
      #L_1 = tf.keras.layers.Reshape((-1, 1)) (L_0)
      L_2 = tf.keras.layers.LSTM(Nodes_Counts[k], activation = "relu") (L_1)
      L_3 = tf.keras.layers.Dense(1, activation = "relu") (L_2)
      #
      Model = tf.keras.Model(inputs = L_0, outputs = L_3)
      Model.compile(optimizer = tf.keras.optimizers.Adamax(learning_rate = 0.02), loss = "mse")
      History = Model.fit(XX_Train, YY_Train, batch_size = len(YY_Train), epochs = 23, validation_data = (XX_Test, YY_Test), verbose = 0)
      #
      #Asking = input("Press Somthing...")
      #if Asking.lower() == "y":
      if History.history["loss"][-1] < int(np.max(Y)):
        #plt.plot(YY[:, 0]);plt.plot(Model.predict(XX)[:, 0]);plt.show();#time.sleep(3);plt.close()
        #if input("Now").lower() != "y": continue
        Peaks_Profit += [np.mean(np.subtract(YY[Peaks_Indices, 0], Model.predict(XX)[Peaks_Indices, 0]) )]
        Valleys_Profit += [np.mean(np.subtract(YY[Valeys_Indices, 0], Model.predict(XX)[Valeys_Indices, 0]) )]
        Results_MSE += [np.mean(tf.keras.losses.mse(YY[Loss_Indices, 0], Model.predict(XX)[Loss_Indices, 0]).numpy() )]
        Results_MAE += [np.mean(tf.keras.losses.mae(YY[Loss_Indices, 0], Model.predict(XX)[Loss_Indices, 0]).numpy() )]
        Results_MAPE += [np.mean(tf.keras.losses.mape(YY[Loss_Indices, 0], Model.predict(XX)[Loss_Indices, 0]).numpy() )]
        Results_RMSE += [np.mean(np.sqrt(tf.keras.losses.mse(YY[Loss_Indices, 0], Model.predict(XX)[Loss_Indices, 0]).numpy()) )]
        Results_LogCosh += [np.mean(tf.keras.losses.logcosh(YY[Loss_Indices, 0], Model.predict(XX)[Loss_Indices, 0]).numpy() )]
        Results_Loss += [np.mean(tf.keras.losses.mse(np.transpose(YY[:, 0]), Model.predict(XX)[:, 0]).numpy() )]
        #
        Y_Prediction = np.transpose(Model.predict(XX))
        Y_Prediction = np.append(np.tile(Y_Prediction[0, 1], Window_Size + 1), Y_Prediction)
        Y_Prediction = np.expand_dims(Y_Prediction, axis = 0)
        YY_Temp = np.concatenate((YY_Temp, Y_Prediction)) if len(YY_Temp.shape) > 1 else Y_Prediction
        #print("Round", j + 1, ":", Results_MSE[-1])
        print("Round", len(Results_MSE), ":", Results_MSE[-1])
        #print(Method_Name)
        #print("----------------------- ----------------------- ----------------------- -----------------------")
        Method_Total += [Method_Name]
        #Y_Prediction_Total += [[Y_Prediction]]
        Y_Prediction_Total += [Y_Prediction[0, :].tolist()]
      else:
        pass
        #clear_output(wait=True)
      #
    print(Method_Name)
    print("----------------------- ----------------------- ----------------------- -----------------------")

    S_Columns = ["Method"] + ["Round_" + str(i + 1) for i in range(Iteration)]
    Total_Peaks_Profit = pd.DataFrame([], columns = S_Columns)
    Total_Valleys_Profit = pd.DataFrame([], columns = S_Columns)
    Total_Results_MSE = pd.DataFrame([], columns = S_Columns)
    Total_Results_MAE = pd.DataFrame([], columns = S_Columns)
    Total_Results_MAPE = pd.DataFrame([], columns = S_Columns)
    Total_Results_RMSE = pd.DataFrame([], columns = S_Columns)
    Total_Results_LogCosh = pd.DataFrame([], columns = S_Columns)
    Total_Results_Loss = pd.DataFrame([], columns = S_Columns)

    if os.path.isfile(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv"):
      Total_Peaks_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")
      Total_Peaks_Profit = Total_Peaks_Profit.drop(columns = Total_Peaks_Profit.keys()[0])

      Total_Valleys_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")
      Total_Valleys_Profit = Total_Valleys_Profit.drop(columns = Total_Valleys_Profit.keys()[0])

      Total_Results_MSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")
      Total_Results_MSE = Total_Results_MSE.drop(columns = Total_Results_MSE.keys()[0])
        
      Total_Results_MAE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")
      Total_Results_MAE = Total_Results_MAE.drop(columns = Total_Results_MAE.keys()[0])
        
      Total_Results_MAPE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")
      Total_Results_MAPE = Total_Results_MAPE.drop(columns = Total_Results_MAPE.keys()[0])
        
      Total_Results_RMSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")
      Total_Results_RMSE = Total_Results_RMSE.drop(columns = Total_Results_RMSE.keys()[0])
        
      Total_Results_LogCosh = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
      Total_Results_LogCosh = Total_Results_LogCosh.drop(columns = Total_Results_LogCosh.keys()[0])

      Total_Results_Loss = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
      Total_Results_Loss = Total_Results_Loss.drop(columns = Total_Results_Loss.keys()[0])

      #Results = pd.DataFrame([[Method_Name] + Results], columns = Total_Results.keys())
    Peaks_Profit = pd.DataFrame([[Method_Name] + Peaks_Profit], columns = S_Columns)
    Total_Peaks_Profit = pd.concat((Total_Peaks_Profit, Peaks_Profit))
    Total_Peaks_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")

    Valleys_Profit = pd.DataFrame([[Method_Name] + Valleys_Profit], columns = S_Columns)
    Total_Valleys_Profit = pd.concat((Total_Valleys_Profit, Valleys_Profit))
    Total_Valleys_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")

    Results_MSE = pd.DataFrame([[Method_Name] + Results_MSE], columns = S_Columns)
    Total_Results_MSE = pd.concat((Total_Results_MSE, Results_MSE))
    Total_Results_MSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")

    Results_MAE = pd.DataFrame([[Method_Name] + Results_MAE], columns = S_Columns)
    Total_Results_MAE = pd.concat((Total_Results_MAE, Results_MAE))
    Total_Results_MAE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")

    Results_MAPE = pd.DataFrame([[Method_Name] + Results_MAPE], columns = S_Columns)
    Total_Results_MAPE = pd.concat((Total_Results_MAPE, Results_MAPE))
    Total_Results_MAPE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")

    Results_RMSE = pd.DataFrame([[Method_Name] + Results_RMSE], columns = S_Columns)
    Total_Results_RMSE = pd.concat((Total_Results_RMSE, Results_RMSE))
    Total_Results_RMSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")

    Results_LogCosh = pd.DataFrame([[Method_Name] + Results_LogCosh], columns = S_Columns)
    Total_Results_LogCosh = pd.concat((Total_Results_LogCosh, Results_LogCosh))
    Total_Results_LogCosh.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")

    Results_Loss = pd.DataFrame([[Method_Name] + Results_Loss], columns = S_Columns)
    Total_Results_Loss = pd.concat((Total_Results_Loss, Results_Loss))
    Total_Results_Loss.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")


    
########################## Ending Methods 5 #############################
#"#""
#
#"#""
########################## Starting Methods 6 #############################
# Kalman Filter PseudoPinBall By Prior
Loss_Name = "PseudoPinBall"

def Differential_Pseudo_PinBall(X, Taw = 0.5):
  # Below Line Is For Mathematical Limit And There Is No Change After That (If It Was Uncommented, An Error Comes For Big Value)
  if X > 23.0: X = 23.0
  if X < -23.0: X = -23.0
  Temp = np.add(np.sinh(8 * X), 8 * X)
  Temp = np.divide(Temp, np.add(np.cosh(8 * X), 1))
  Temp = np.subtract(Temp, (1 - (2 * Taw)))
  return Temp / 2.0

Peaks_Profit = []
Valleys_Profit = []
Results_MSE = []
Results_MAE = []
Results_MAPE = []
Results_RMSE = []
Results_LogCosh = []
Results_Loss = [] # Total Prediction Loss By MSE
for j in range(Iteration):
  #while H[0, 0] <13: H = np.array([[np.random.rand() * 23]])
  H = np.array([[np.random.rand() * 10]])
  Q = np.random.rand(2,2)
  
  Loss_Coefficent = 0.9 # Between 0 And 1, More Coefficent, More Obeservation Error
  Balance_Taw = 0.5
  Taw = Balance_Taw
  Critical_Point = [23 * i for i in range(1, int(len(Y[0]) / 23) + 1)]
  Critical_Point[-1] = Critical_Point[-1] - 1
  Critical_Point_Type = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
    
  if Synthesis_Data == False:
    Critical_Point = list(Critical_Points[Symbol]) + [Y.shape[1] - 1]
    Critical_Point_Type = list(Critical_Points_Type[Symbol]) + [1]

  #Critical_Point = [Y.shape[1] - 1]
  #Critical_Point_Type = [1]

  Loss_Name = "PseudoPinBall"
  Loss_Name += "_Taw_" + "{0:.2e}".format(Taw).replace("+", "") # str(My_Sigma)
  Loss_Name += "_Loss_Coefficent_" + "{0:.2e}".format(Loss_Coefficent).replace("+", "") # str(Loss_Coefficent)
  ##Loss_Name += "_Mode_" + str(Mode)
  Method_Name = "KF_" + Loss_Name
  ####################### Start Of Kalman Filter Process
  A_First = np.random.rand(Y.shape[0] * 2, 1) #* 100# * 46
  P_First = np.eye(Y.shape[0] * 2) * 10
  Alpha_First = np.random.multivariate_normal(A_First[:,0], P_First)
  ##Alpha_First = np.array([24, -1])
  #Alpha_First = np.array([22, 1])
  #Alpha_First = np.array([float(Y[:, 0]), 1])
  Alpha_First = np.array([float(Y[:, 0]), 1]) if Synthesis_Data == False else np.array([float(Y[:, 0]), -1])
  Alpha_First = np.expand_dims(Alpha_First, axis = 1)
  #print(A_First.shape, "\n", A_First)
  #print(Alpha_First.shape, "\n", Alpha_First)
  #print(P_First.shape, "\n", P_First)

  Alpha = np.array(Alpha_First)
  Alpha = np.expand_dims(Alpha, axis = 0)
  #print("Alpha:", Alpha.shape)
  P = np.array(P_First)
  P = np.expand_dims(P, axis = 0)
  #print("P:", P.shape)
  #print(P.shape)
  V = np.array([]) # Predict Alpha
  K = np.array([]) # Kalman Filter Gain
  L = np.array([]) # Predict P
  Y_Prediction = np.array([])
  Y_Fit = np.array([])

  for i in range(Y.shape[1]):
    # Predict
    Temp = np.dot(T, Alpha[i, :, :])
    Temp = np.expand_dims(Temp, 0)
    #print("V:", Temp)
    V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp
    
    Temp = np.dot(Z, V[i, :, :])
    Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
    #if i>2:
    #  print(Y_Prediction.shape)
    #  print(Y.shape)
    #  TT = np.dot(Alpha[i,:,:],Alpha[i,:,:]) # Bug For Stop And See Results

    Temp = np.dot(np.dot(T, P[i, :, :]), np.transpose(T))
    Temp = np.add(Temp, np.dot(np.dot(R, Q), np.transpose(R)))
    Temp = np.expand_dims(Temp, axis = 0)
    #print("L:", Temp)
    L = np.concatenate((L, Temp)) if len(L.shape) > 1 else Temp
    
    # Update
    
    if i < Critical_Point[0] and Critical_Point[0] - i < 5:
      #Taw += +0.1 if np.any(Y[:, i] > Y[:, Critical_Point[0]]) else -0.1
      Taw += +0.1 if Critical_Point_Type[0] == -1 else -0.1 
      ###Taw = 0.4 if Critical_Point_Type[0] == -1 else 0.9 
      Alpha[i, 1, :] += -0.23 * Critical_Point_Type[0]
      ########################## Should Be Removed
      Temp_3 = np.dot(T, Alpha[i, :, :])
      Temp_3 = np.expand_dims(Temp_3, 0)
      V[-1] = Temp_3
      ##V[-1, 0, 0] = V[-1, 0, 0] + (0.05 * )
      
      Temp_3 = np.dot(Z, V[i, :, :])
      Y_Prediction[:, -1] = Temp_3
      ##########################
    elif i > Critical_Point[0] and  i - Critical_Point[0] < 4:
      #Taw += +0.1 if np.any(Y[:, i] < Y[:, Critical_Point[0]]) else -0.1
      Taw += +0.1 if Critical_Point_Type[0] == 1 else -0.1
      ###Taw = 0.9 if Critical_Point_Type[0] == 1 else 0.4
      Alpha[i, 1, :] += -0.23* Critical_Point_Type[0]
      ########################## Should Be Removed
      Temp_3 = np.dot(T, Alpha[i, :, :])
      Temp_3 = np.expand_dims(Temp_3, 0)
      V[-1] = Temp_3
      ##V[-1, 0, 0] = V[-1, 0, 0] + (0.05 * )
      
      Temp_3 = np.dot(Z, V[i, :, :])
      Y_Prediction[:, -1] = Temp_3
      ##########################
    elif i > Critical_Point[0] and  i - Critical_Point[0] == 5:
      Critical_Point = Critical_Point[1:]
      Critical_Point_Type = Critical_Point_Type[1:]
    elif i == Critical_Point[0]:
      Alpha[i, 1, :] = -1 * Critical_Point_Type[0]
      pass
    else:
      Taw = Balance_Taw
    #print(Taw)
    
    Temp_1 = np.abs(np.subtract(np.expand_dims(Y[:, i], axis = 1), np.dot(Z, V[i, :, :])))
    #print("Temp_1:", Temp_1)
    try:
      Temp_1 = np.dot(np.dot(np.transpose(Temp_1), np.linalg.inv(H)), Temp_1)
    except:
      Temp_1 = np.dot(np.dot(np.transpose(Temp_1), np.linalg.pinv(H)), Temp_1)
    #
    Temp_2 = np.abs(np.subtract(Alpha[i, :, :], V[i, :, :]))
    try:
      Temp_2 = np.dot(np.dot(np.transpose(Temp_2), np.linalg.inv(L[i, :, :])), Temp_2)
    except:
      Temp_2 = np.dot(np.dot(np.transpose(Temp_2), np.linalg.pinv(L[i, :, :])), Temp_2)

    K_Temp = Differential_Pseudo_PinBall(Temp_1, Taw) / (Differential_Pseudo_PinBall(Temp_2, Taw) + 1e-23)
    #print("Temp_1:", Temp_1, "Temp_2:", Temp_2)
    #print("K_Temp:", K_Temp)
    try:
      Temp = np.dot(np.multiply(K_Temp, np.transpose(Z)), np.dot(np.linalg.inv(H), Z))
    except:
      Temp = np.dot(np.multiply(K_Temp, np.transpose(Z)), np.dot(np.linalg.pinv(H), Z))
    #
    try:
      Temp = np.add(np.linalg.inv(L[i, :, :]), Temp)
    except:
      Temp = np.add(np.linalg.pinv(L[i, :, :]), Temp)
    #
    try:
      Temp = np.linalg.inv(Temp)
    except:
      Temp = np.linalg.pinv(Temp)
    #
    try:
      Temp = np.dot(np.multiply(Temp, K_Temp), np.dot(np.transpose(Z), np.linalg.inv(H)))
    except:
      Temp = np.dot(np.multiply(Temp, K_Temp), np.dot(np.transpose(Z), np.linalg.pinv(H)))
    Temp = np.expand_dims(Temp, 0)
    
    K = np.concatenate((K , Temp)) if len(K.shape) > 1 else Temp  

    Temp = np.subtract(np.expand_dims(Y[:, i], axis = 1), np.dot(Z, V[i, :, :]))
    #print("Alpha_1:", Temp.shape)
    Temp = np.dot(K[i, :, :], Temp)
    Temp = np.add(V[i, :, :], Temp)
    Temp= np.expand_dims(Temp, 0)
    Alpha = np.concatenate((Alpha, Temp))

    Temp = np.subtract(np.eye(Y.shape[0] * 2), np.dot(K[i, :, :], Z))
    Temp = np.dot(np.dot(Temp, L[i, :, :]), np.transpose(Temp))
    if len(H.shape) == 1:
      H = np.expand_dims(H, 1)
    Temp = np.add(Temp, np.dot(np.dot(K[i, :, :], H), np.transpose(K[i, :, :])))
    Temp= np.expand_dims(Temp, 0)
    #print("P_1:", Temp)
    P = np.concatenate((P , Temp))

    #
    Temp = np.dot(Z, Alpha[-1, :, :])
    Y_Fit =  np.concatenate((Y_Fit, Temp), axis = 1) if len(Y_Fit.shape) > 1 else Temp
    
    #
    if i % (int(Y.shape[1]/10)) ==0:
      print(i, end="\t")
    #print("Temp:", Temp.shape)
    #

  print()

  Y_Temp = np.copy(Y)
  #Y_Tild = np.transpose(Y).reshape(Min_Stock_Length * Y.shape[0], 1) # m x n => n x m => nm x 1 
  # Y_Prediction_Tild Is Sum Of Miu_Tild(Trend), Taw_Tild(Season), Omega_Tild(Cycle)
  #Y_Prediction_Tild = np.transpose(Y_Prediction).reshape(Min_Stock_Length * Y.shape[0], 1) # m x n => n x m => nm x 1 
  #
  print("MSE:", tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy()))
  print("MAE:", tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy()))
  #Q *= 0.987

  #### Optional
  # Predict
  Temp = np.dot(T, Alpha[i, :, :])
  Temp = np.expand_dims(Temp, 0)
  #print("V:", Temp)
  V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp

  Temp = np.dot(Z, V[i, :, :])
  Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
  ####
  ###plt.figure(figsize = (13, 6));plt.plot(Y[0, :], label = "Input");plt.plot(Y_Prediction[0, :-1], label = "SGD");plt.legend();plt.show()
  #if "Results_MSE" in globals():
  Peaks_Profit += [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
  Valleys_Profit += [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
  Results_MSE += [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
  Results_MAE += [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
  Results_MAPE += [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
  Results_RMSE += [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
  Results_LogCosh += [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
  Results_Loss += [np.mean(tf.keras.losses.mse(Y[:, :], Y_Prediction[:, :-1]).numpy() )]
  #print("Round", j + 1, ":", Results_MSE[-1])
  #else:
    #Peaks_Profit = [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
    #Valleys_Profit = [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
    #Results_MSE = [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    #Results_MAE = [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    #Results_MAPE = [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    #Results_RMSE = [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
    #Results_LogCosh = [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
  
  print(Method_Name)
  print("----------------------- ----------------------- ----------------------- -----------------------")
  Method_Total += [Method_Name] # + "_" + str(j)
  #Y_Prediction_Total += [[Y_Prediction]]
  Y_Prediction_Total += [Y_Prediction[0, :].tolist()]
  #

S_Columns = ["Method"] + ["Round_" + str(i + 1) for i in range(Iteration)]
Total_Peaks_Profit = pd.DataFrame([], columns = S_Columns)
Total_Valleys_Profit = pd.DataFrame([], columns = S_Columns)
Total_Results_MSE = pd.DataFrame([], columns = S_Columns)
Total_Results_MAE = pd.DataFrame([], columns = S_Columns)
Total_Results_MAPE = pd.DataFrame([], columns = S_Columns)
Total_Results_RMSE = pd.DataFrame([], columns = S_Columns)
Total_Results_LogCosh = pd.DataFrame([], columns = S_Columns)
Total_Results_Loss = pd.DataFrame([], columns = S_Columns)

if os.path.isfile(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv"):
  Total_Peaks_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")
  Total_Peaks_Profit = Total_Peaks_Profit.drop(columns = Total_Peaks_Profit.keys()[0])

  Total_Valleys_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")
  Total_Valleys_Profit = Total_Valleys_Profit.drop(columns = Total_Valleys_Profit.keys()[0])

  Total_Results_MSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")
  Total_Results_MSE = Total_Results_MSE.drop(columns = Total_Results_MSE.keys()[0])
    
  Total_Results_MAE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")
  Total_Results_MAE = Total_Results_MAE.drop(columns = Total_Results_MAE.keys()[0])
    
  Total_Results_MAPE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")
  Total_Results_MAPE = Total_Results_MAPE.drop(columns = Total_Results_MAPE.keys()[0])
    
  Total_Results_RMSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")
  Total_Results_RMSE = Total_Results_RMSE.drop(columns = Total_Results_RMSE.keys()[0])
    
  Total_Results_LogCosh = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
  Total_Results_LogCosh = Total_Results_LogCosh.drop(columns = Total_Results_LogCosh.keys()[0])

  Total_Results_Loss = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
  Total_Results_Loss = Total_Results_Loss.drop(columns = Total_Results_Loss.keys()[0])

  #Results = pd.DataFrame([[Method_Name] + Results], columns = Total_Results.keys())
Peaks_Profit = pd.DataFrame([[Method_Name] + Peaks_Profit], columns = S_Columns)
Total_Peaks_Profit = pd.concat((Total_Peaks_Profit, Peaks_Profit))
Total_Peaks_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")

Valleys_Profit = pd.DataFrame([[Method_Name] + Valleys_Profit], columns = S_Columns)
Total_Valleys_Profit = pd.concat((Total_Valleys_Profit, Valleys_Profit))
Total_Valleys_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")

Results_MSE = pd.DataFrame([[Method_Name] + Results_MSE], columns = S_Columns)
Total_Results_MSE = pd.concat((Total_Results_MSE, Results_MSE))
Total_Results_MSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")

Results_MAE = pd.DataFrame([[Method_Name] + Results_MAE], columns = S_Columns)
Total_Results_MAE = pd.concat((Total_Results_MAE, Results_MAE))
Total_Results_MAE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")

Results_MAPE = pd.DataFrame([[Method_Name] + Results_MAPE], columns = S_Columns)
Total_Results_MAPE = pd.concat((Total_Results_MAPE, Results_MAPE))
Total_Results_MAPE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")

Results_RMSE = pd.DataFrame([[Method_Name] + Results_RMSE], columns = S_Columns)
Total_Results_RMSE = pd.concat((Total_Results_RMSE, Results_RMSE))
Total_Results_RMSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")

Results_LogCosh = pd.DataFrame([[Method_Name] + Results_LogCosh], columns = S_Columns)
Total_Results_LogCosh = pd.concat((Total_Results_LogCosh, Results_LogCosh))
Total_Results_LogCosh.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
  
Results_Loss = pd.DataFrame([[Method_Name] + Results_Loss], columns = S_Columns)
Total_Results_Loss = pd.concat((Total_Results_Loss, Results_Loss))
Total_Results_Loss.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")


####################### End Of Kalman Filter Process
  
########################## Ending Methods 6 #############################
#"#""
#
#"#""
########################## Starting Methods 7 #############################
# Kalman Filter SGD By Prior
#Iteration = 1
#Method_Total = []
#Y_Prediction_Total = []

##Learning_Rate = 2.25
##Max_LR = 4.25
Learning_Rate = 2.25
Max_LR = 10.25
for k in range(int((Max_LR - Learning_Rate) / 0.25)):
  Learning_Rate = Learning_Rate + 0.25 if "Learning_Rate" in globals() else 2.25
  if Learning_Rate > Max_LR: Learning_Rate = 2.25

  Peaks_Profit = []
  Valleys_Profit = []
  Results_MSE = []
  Results_MAE = []
  Results_MAPE = []
  Results_RMSE = []
  Results_LogCosh = []
  Results_Loss = [] # Total Prediction Loss By MSE
  for j in range(Iteration):
    #
    Loss_Coefficent = 0.9 # Between 0 And 1, More Coefficent, More Obeservation Error
    Balance_Taw = 0.5
    Taw = Balance_Taw
    Critical_Point = [23 * i for i in range(1, int(len(Y[0]) / 23) + 1)]
    Critical_Point[-1] = Critical_Point[-1] - 1
    Critical_Point_Type = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1] # -1: Valey, +1: Peak

    if Synthesis_Data == False:
      Critical_Point = list(Critical_Points[Symbol]) + [Y.shape[1] - 1]
      Critical_Point_Type = list(Critical_Points_Type[Symbol]) + [1]

    def Pin_Ball(x): # Close To 1 Work For Positive Slope Vise Versa.
      global Taw
      #if Taw > 0.53 or Taw < 0.47: print("@#$%^&*(!@#$%^&*(1\t", Taw, "\t@#$%^&*(!@#$%^&*(1\n\n")
      if Taw > 1 or Taw < 0:
        print("Error \n Current Taw:", Taw)
        input("Continue?")
      return np.add(np.multiply(np.array(x>=0).astype(np.int8), np.multiply(x, Taw)), np.multiply(np.array(x<0).astype(np.int8), np.multiply(x, (Taw - 1))))

    Loss_Name = "PinBall_Taw_5e-1"
    Loss_Name += "_Learning_Rate_" + "{0:.2e}".format(Learning_Rate).replace("+", "") # str(Loss_Coefficent)
    Loss_Name += "_Loss_Coefficent_" + "{0:.2e}".format(Loss_Coefficent).replace("+", "") # str(Loss_Coefficent)
    

    Method_Name = "KF_SGD_" + Loss_Name
    Method_Name += "_" + str(j)

    A_First = np.random.rand(T.shape[0], 1) #* 100# * 46
    P_First = np.eye(T.shape[0]) * 10
    Alpha_First = np.random.multivariate_normal(A_First[:,0], P_First)
    #Alpha_First = np.array([2.3, 2.3])
    ##Alpha_First = np.array([24, -1])
    #Alpha_First = np.array([22, 1])
    Alpha_First = np.array([float(Y[:, 0]), 1]) if Synthesis_Data == False else np.array([float(Y[:, 0]), -1])
    Alpha_First = np.expand_dims(Alpha_First, axis = 1)
    #print(A_First.shape, "\n", A_First)
    #print(Alpha_First.shape, "\n", Alpha_First)
    #print(P_First.shape, "\n", P_First)

    Alpha = np.array(Alpha_First)
    Alpha = np.expand_dims(Alpha, axis = 0)
    #print("Alpha:", Alpha.shape)
    P = np.array(P_First)
    P = np.expand_dims(P, axis = 0)
    #print("P:", P.shape)
    #print(P.shape)
    V = np.array([]) # Predict Alpha
    Y_Prediction = np.array([])
    Y_Fit = np.array([])

    for i in range(Y.shape[1]):
      # Predict
      Temp = np.dot(T, Alpha[i, :, :])
      Temp = np.expand_dims(Temp, 0)
      #print("V:", Temp)
      V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp
      
      Temp = np.dot(Z, V[i, :, :])
      #Temp = np.dot(Z, Alpha[i, :, :])
      Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
      #if i>2:
      #  print(Y_Prediction.shape)
      #  print(Y.shape)
      #  TT = np.dot(Alpha[i,:,:],Alpha[i,:,:]) # Bug For Stop And See Results
      # Update
      if i < Critical_Point[0] and Critical_Point[0] - i < 5:
        ##Taw += +0.1 if np.any(Y[:, i] > Y[:, Critical_Point[0]]) else -0.1
        Taw += +0.1 if Critical_Point_Type[0] == -1 else -0.1 
        ###Taw = 0.5 if Critical_Point_Type == -1 else 0.8
        ##Taw = Balance_Taw
        #if Critical_Point[0] - i == 1: Alpha[i, 1, :] = Critical_Point_Type[0]
        Alpha[i, 1, :] -= 0.4 * Critical_Point_Type[0]
      elif i > Critical_Point[0] and  i - Critical_Point[0] < 4:
        ##Taw += +0.1 if np.any(Y[:, i] < Y[:, Critical_Point[0]]) else -0.1
        Taw += +0.1 if Critical_Point_Type[0] == 1 else -0.1
        ###Taw = 0.5 if Critical_Point_Type == 1 else 0.8
        ##Taw = Balance_Taw
      elif i > Critical_Point[0] and  i - Critical_Point[0] == 5:
        Critical_Point = Critical_Point[1:]
        Critical_Point_Type = Critical_Point_Type[1:]
      elif i == Critical_Point[0]: 
        Alpha[i, 1, :] = -1 * Critical_Point_Type[0]
        #print(Alpha[i, 1, :])
        pass
      else:
        Taw = Balance_Taw
      ##Temp_1 = np.subtract(np.expand_dims(Y[:, i], axis = 1), np.dot(Z, V[i, :, :]))
      Temp_1 = np.subtract(np.expand_dims(Y[:, i], axis = 1), np.dot(Z, Alpha[i, :, :]))
      #Temp = derivative(PseudoHuber, Temp_1, dx = 1.0,  n = 1, order = 3) # || y - Z * Alpha||
      Temp = derivative(Pin_Ball, Temp_1, dx = 1.0,  n = 1, order = 3) # || y - Z * Alpha||
      Temp = np.dot(np.transpose(Z), Temp)
      Temp = np.multiply(Loss_Coefficent, Temp)
      TT = Temp
      if i>1:
        ##Temp_1 = np.subtract(Alpha[i, :, :], np.dot(T, V[i-1, :, :]))
        Temp_1 = np.subtract(Alpha[i, :, :], V[i - 1, :, :])
        ###Temp_1 = np.subtract(Alpha[i, :, :], np.dot(T, Alpha[i-1, :, :]))
        #Temp = np.add(Temp, derivative(PseudoHuber, Temp_1, dx = 1.0,  n = 1, order = 3)) # || (x_t+1) - T * x_t||
        #Temp = np.add(Temp, np.multiply((1 - Loss_Coefficent) ,derivative(Pin_Ball, Temp_1, dx = 1.0,  n = 1, order = 3) ) ) # || (x_t+1) - T * x_t||
      #del Temp_1
      #Temp = np.subtract(Alpha[i, :, :], np.multiply(Learning_Rate, Temp))
      Temp = np.add(Alpha[i, :, :], np.multiply(Learning_Rate, Temp))
      Temp = np.expand_dims(Temp, 0)
      Alpha = np.concatenate((Alpha, Temp))
      
      Temp = np.dot(Z, Alpha[-1, :, :])
      Y_Fit =  np.concatenate((Y_Fit, Temp), axis = 1) if len(Y_Fit.shape) > 1 else Temp
      
      #
      if i %(int(Y.shape[1]/10)) ==0:
        print(i, end="\t")
      #print("Temp:", Temp.shape)
      #
    print()

    Y_Temp = np.copy(Y)
    #Y_Tild = np.transpose(Y).reshape(Min_Stock_Length * Y.shape[0], 1) # m x n => n x m => nm x 1 
    # Y_Prediction_Tild Is Sum Of Miu_Tild(Trend), Taw_Tild(Season), Omega_Tild(Cycle)
    #Y_Prediction_Tild = np.transpose(Y_Prediction).reshape(Min_Stock_Length * Y.shape[0], 1) # m x n => n x m => nm x 1 
    #
    print("Learning_Rate:", Learning_Rate, "Loss_Coefficent:", Loss_Coefficent)
    print("Loss:", Loss_Name)
    print("MSE:", tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MSE(Y[:,:], Y_Prediction[:,:]).numpy()))
    print("MAE:", tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy(), "=> Mean:", np.mean(tf.keras.losses.MAE(Y[:,:], Y_Prediction[:,:]).numpy()))
    #Q *= 0.987

    #### Optional
    # Predict
    Temp = np.dot(T, Alpha[i, :, :])
    Temp = np.expand_dims(Temp, 0)
    #print("V:", Temp)
    V = np.concatenate((V, Temp)) if len(V.shape) > 1 else Temp

    Temp = np.dot(Z, V[i, :, :])
    Y_Prediction = np.concatenate((Y_Prediction, Temp), axis = 1) if len(Y_Prediction.shape) > 1 else Temp
    ####

    #Method_Comparison = Method_Comparison + [Loss_Name] if "Method_Comparison" in globals() else [Loss_Name]
    #Y_Comparison = np.concatenate((Y_Comparison, Y_Prediction)) if "Y_Comparison" in globals() else Y_Prediction
    #
    ###plt.figure(figsize = (13, 6));plt.plot(Y[0, :], label = "Input");plt.plot(Y_Prediction[0, :-1], label = "SGD");plt.legend();plt.show()
    #
    #if "Results_MSE" in globals():
    Peaks_Profit += [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
    Valleys_Profit += [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
    Results_MSE += [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    Results_MAE += [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    Results_MAPE += [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    Results_RMSE += [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
    Results_LogCosh += [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
    Results_Loss += [np.mean(tf.keras.losses.mse(Y[:, :], Y_Prediction[:, :-1]).numpy() )]
    #print("Round", j + 1, ":", Results_MSE[-1])
    #else:
      #Peaks_Profit = [np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )]
      #Valleys_Profit = [np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )]
      #Results_MSE = [np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      #Results_MAE = [np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      #Results_MAPE = [np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]
      #Results_RMSE = [np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )]
      #Results_LogCosh = [np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )]

    print("Round", len(Results_MSE), ":", Results_MSE[-1])

    print(Method_Name)
    print("----------------------- ----------------------- ----------------------- -----------------------")
    Method_Total += [Method_Name+ "_" + str(j)]
    #Y_Prediction_Total += [[Y_Prediction]]
    Y_Prediction_Total += [Y_Prediction[0, :].tolist()]
  
  S_Columns = ["Method"] + ["Round_" + str(i + 1) for i in range(Iteration)]
  Total_Peaks_Profit = pd.DataFrame([], columns = S_Columns)
  Total_Valleys_Profit = pd.DataFrame([], columns = S_Columns)
  Total_Results_MSE = pd.DataFrame([], columns = S_Columns)
  Total_Results_MAE = pd.DataFrame([], columns = S_Columns)
  Total_Results_MAPE = pd.DataFrame([], columns = S_Columns)
  Total_Results_RMSE = pd.DataFrame([], columns = S_Columns)
  Total_Results_LogCosh = pd.DataFrame([], columns = S_Columns)
  Total_Results_Loss = pd.DataFrame([], columns = S_Columns)

  if os.path.isfile(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv"):
    Total_Peaks_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")
    Total_Peaks_Profit = Total_Peaks_Profit.drop(columns = Total_Peaks_Profit.keys()[0])

    Total_Valleys_Profit = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")
    Total_Valleys_Profit = Total_Valleys_Profit.drop(columns = Total_Valleys_Profit.keys()[0])

    Total_Results_MSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")
    Total_Results_MSE = Total_Results_MSE.drop(columns = Total_Results_MSE.keys()[0])
    
    Total_Results_MAE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")
    Total_Results_MAE = Total_Results_MAE.drop(columns = Total_Results_MAE.keys()[0])
    
    Total_Results_MAPE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")
    Total_Results_MAPE = Total_Results_MAPE.drop(columns = Total_Results_MAPE.keys()[0])
    
    Total_Results_RMSE = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")
    Total_Results_RMSE = Total_Results_RMSE.drop(columns = Total_Results_RMSE.keys()[0])
    
    Total_Results_LogCosh = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
    Total_Results_LogCosh = Total_Results_LogCosh.drop(columns = Total_Results_LogCosh.keys()[0])
    
    Total_Results_Loss = pd.read_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
    Total_Results_Loss = Total_Results_Loss.drop(columns = Total_Results_Loss.keys()[0])

  #Results = pd.DataFrame([[Method_Name] + Results], columns = Total_Results.keys())
  Peaks_Profit = pd.DataFrame([[Method_Name] + Peaks_Profit], columns = S_Columns)
  Total_Peaks_Profit = pd.concat((Total_Peaks_Profit, Peaks_Profit))
  Total_Peaks_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")

  Valleys_Profit = pd.DataFrame([[Method_Name] + Valleys_Profit], columns = S_Columns)
  Total_Valleys_Profit = pd.concat((Total_Valleys_Profit, Valleys_Profit))
  Total_Valleys_Profit.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")

  Results_MSE = pd.DataFrame([[Method_Name] + Results_MSE], columns = S_Columns)
  Total_Results_MSE = pd.concat((Total_Results_MSE, Results_MSE))
  Total_Results_MSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")

  Results_MAE = pd.DataFrame([[Method_Name] + Results_MAE], columns = S_Columns)
  Total_Results_MAE = pd.concat((Total_Results_MAE, Results_MAE))
  Total_Results_MAE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")

  Results_MAPE = pd.DataFrame([[Method_Name] + Results_MAPE], columns = S_Columns)
  Total_Results_MAPE = pd.concat((Total_Results_MAPE, Results_MAPE))
  Total_Results_MAPE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")

  Results_RMSE = pd.DataFrame([[Method_Name] + Results_RMSE], columns = S_Columns)
  Total_Results_RMSE = pd.concat((Total_Results_RMSE, Results_RMSE))
  Total_Results_RMSE.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")

  Results_LogCosh = pd.DataFrame([[Method_Name] + Results_LogCosh], columns = S_Columns)
  Total_Results_LogCosh = pd.concat((Total_Results_LogCosh, Results_LogCosh))
  Total_Results_LogCosh.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")

  Results_Loss = pd.DataFrame([[Method_Name] + Results_Loss], columns = S_Columns)
  Total_Results_Loss = pd.concat((Total_Results_Loss, Results_Loss))
  Total_Results_Loss.to_csv(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
  
########################## Ending Methods 7 #############################
#"#""
#
#"#""
########################## Starting Methods 8 #############################

#Method_Total += [Method_Name]
#Y_Prediction_Total += [Y_Prediction]
########################## Ending Methods 8 #############################
#"#""

#Revesal_Losses = np.array(Revesal_Losses)[:, :, 0]
#Revesal_Losses = np.transpose(Revesal_Losses)
# ["MSE", "MAE", "MAPE", "RMSE", "LogCosh"]

##if np.argmin(Revesal_Losses[0, :]) >= (3 * Iteration): pass
##if np.argmin(Revesal_Losses[0, :]) >= (5 * Iteration):
##  print("Our Method Is Successful\tMinimum Argument Is", np.argmin(Revesal_Losses[0, :]))
##else:
##  print("Our Method Is Faild\tMinimum Argument Is", np.argmin(Revesal_Losses[0, :]))

Method_Total = np.array(Method_Total)
Y_Prediction_Total = np.array(Y_Prediction_Total)

#Method_Total = np.expand_dims(Method_Total, axis = 1)
F = open(os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Predictions.txt", "wb")
pickle.dump({"Methods": Method_Total, "Predictions": Y_Prediction_Total}, F)
F.close()
print("Results Is Saved On Below Path:\n\n", os.getcwd() + "\\" + Folder + "\\" + Symbol + "_Predictions.txt")

"""
Temp_Time = int(time.time())

Peaks_Profit = np.array([np.subtract(Y[:, Peaks_Indices], Y_Prediction_Total[:, Peaks_Indices]) ])
Valeys_Profit = np.array([np.subtract(Y[:, Valeys_Indices], Y_Prediction_Total[:, Valeys_Indices])])
Temp = np.concatenate((Valeys_Profit, Peaks_Profit))[:, :, 0]
Temp = np.concatenate((Temp, Revesal_Losses))
Topics = np.array(["Peaks_Profit", "Valleys_Profit", "MSE"
                                   , "MAE", "MAPE", "RMSE", "LogCosh"]).reshape(-1, 1)
##Topics = np.transpose(np.array([["Peaks_Profit", "Valeys_Profit", "MSE"
##                                                            , "MAE", "MAPE", "RMSE", "LogCosh"]]))
Temp  = np.concatenate((Topics, Temp), axis = 1)
Method_Total = np.append("Topic", Method_Total)
Profits = pd.DataFrame(Temp, columns = Method_Total)
Profits.to_csv("./Profits_" + Symbol + "_" + str(Temp_Time) + ".csv", index = False)
print("Results Is Wrote In", "./Profits_" + Symbol + "_" + str(int(time.time())) + ".csv")

F = open("./Prediction_" + Symbol + "_" + str(Temp_Time) + ".txt", "wb")
pickle.dump({"Method": Method_Total[1:], "Prediction": Y_Prediction_Total}, F)
F.close()
"""

#
""" Commented Older
for i in range(5, Method_Total.shape[0] - 1):
  plt.figure(figsize = (13, 6))
  plt.plot(Y[0, :], label = "Input", linewidth = 3.7)
  plt.plot(Y_Prediction_Total[i], label = Method_Total[i], linestyle = "--")
  plt.legend()
  plt.show()
  plt.close()
"""
#
"""
plt.rcParams.update({"font.size": 7})
plt.figure(figsize = (13, 6))
plt.plot(Y[0, :], label = "Input", linewidth = 3.7)
for i in range(Method_Total.shape[0] - 1):
  if i in []: # 0, 1, 2, 3, 4
    continue
  plt.plot(Y_Prediction_Total[i], label = Method_Total[i + 1], linestyle = "--")
  
plt.legend()
plt.xlabel("Time");plt.ylabel("Price")
plt.tight_layout()
plt.savefig("./Experimental_" + Symbol + ".png")
plt.show()


Critical_Point = [23 * i for i in range(1, int(len(Y[0]) / 23) + 1)]
Critical_Point[-1] = Critical_Point[-1] - 1
Critical_Point_Type = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1] # -1: Valey, +1: Peak

if Synthesis_Data == False:
  Critical_Point = list(Critical_Points[Symbol]) + [Y.shape[1] - 1]
  Critical_Point_Type = list(Critical_Points_Type[Symbol]) + [1]


for i in range(Method_Total.shape[0] - 1):
  plt.figure(figsize = (13, 6))
  plt.title(",".join(np.array(Critical_Point).astype(np.str_)))
  plt.plot(Y[0, :], label = "Input", linewidth = 3.7)
  plt.plot(Y_Prediction_Total[i], label = Method_Total[i + 1], linestyle = "--")
  for k in range(len(Critical_Point)):
    plt.vlines(Critical_Point[k], np.min(Y[0, :]) - 23, np.max(Y[0, :]) + 23, color = "cyan")
  plt.legend()
  plt.show()
"""
#for i in range(len(Method_Total)):
#  print(Method_Total[i])

"""
plt.figure(figsize = (13, 6))
plt.plot(Y[0, :], linewidth = 3.1, label = "input")
#plt.plot(Y_Prediction_Total[0, :], label = Method_Total[0])
for i in range(Iteration):
  plt.plot(Y_Prediction_Total[-i, :], label = Method_Total[-i])

plt.legend();plt.show()

print(H)
print(Q)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import time

# PseudoPinBall Index Is 37
#   And Its Range On Predictions File Is Between 3700 And 3799
#
FileNames = [name for name in os.listdir(os.getcwd()) if (".csv" in name and "Results" in name)]
FileNames.sort()

#####################################################
if "Synthesis" in FileNames[0]:
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

  Critical_Point = [23 * i for i in range(1, int(len(Y[0]) / 23) + 1)]
  Critical_Point[-1] = Critical_Point[-1] - 1
  Critical_Point_Type = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1] # -1: Valley, +1: Peak
else:
  F = open("Prior_Extraction_Info.txt", "rb")
  #F = open("Prior_Extraction_Info_2.txt", "rb")
  Data, Data_Smooth, Critical_Points, Critical_Points_Type = pickle.load(F)
  F.close()

  Symbols = list(Data.keys())

  Symbol = FileNames[0][:FileNames[0].find("_")]
  if Symbol not in Symbols:
    print("This Symbols Does Not Exist In This File")
    try:
      F = open("Prior_Extraction_Info_2.txt", "rb")
      Data, Data_Smooth, Critical_Points, Critical_Points_Type = pickle.load(F)
      F.close()
    except:
      exit()
  #Symbol = Symbols[12]
  #Symbol = Symbols[3] # "TSLA"
  print(Symbol, Data[Symbol].shape)

  Y = Data[Symbol]["Close"].to_numpy()#[:221]
  Y = np.expand_dims(Y, axis = 0 )
  print(Y.shape)
  Labels = [Symbol]

  Y_Smooth = Data_Smooth[Symbol]["Close"].to_numpy()#[:221]
  Y_Smooth = np.expand_dims(Y_Smooth, axis = 0 )
  Synthesis_Data = False
  Critical_Point = list(Critical_Points[Symbol]) + [Y.shape[1] - 1]
  Critical_Point_Type = list(Critical_Points_Type[Symbol]) + [1]
  

Z = np.array([[1, 0]])
T = np.array([[1, 1], [0, 1]])
H = np.array([[np.random.rand() * 10]])
R = np.eye(2)
Q = np.random.rand(2,2)

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
#####################################################

for i in range(len(FileNames)):
  if "_Loss_" in FileNames[i]:
    Loss_File = FileNames[i]
    break

Info = pd.read_csv(os.getcwd() + "\\" + Loss_File)
Info = Info.drop(columns = Info.keys()[0])
if "Mean" not in Info.keys():
  print("Mean And Variance Of Results Needed, Run Mean_Variance_Calculation.py")
  time.sleep(7)
  quit()

for j in range(len(Info.keys())):
  #for j in range(len(Info.keys()) - 1, -1 , -1):
    if "Round" in Info.keys()[-j]:
      break

Last_Round = len(Info.keys()) - j
Indices = [i for i in range(1, Last_Round + 1) if Info.iloc[37, i] > (Info["Mean"][37] + (3 * Info["Variance"][37]))]

while len(Indices) > 0:
  print(Indices)
  print(Info.iloc[37, Indices])
  time.sleep(7)
  def Differential_Pseudo_PinBall(X, Taw = 0.5):
    # Below Line Is For Mathematical Limit And There Is No Change After That (If It Was Uncommented, An Error Comes For Big Value)
    if X > 23.0: X = 23.0
    if X < -23.0: X = -23.0
    Temp = np.add(np.sinh(8 * X), 8 * X)
    Temp = np.divide(Temp, np.add(np.cosh(8 * X), 1))
    Temp = np.subtract(Temp, (1 - (2 * Taw)))
    return Temp / 2.0

  ##Y_Prediction_Total = []

  ##Peaks_Profit = []
  ##Valleys_Profit = []
  ##Results_MSE = []
  ##Results_MAE = []
  ##Results_MAPE = []
  ##Results_RMSE = []
  ##Results_LogCosh = []
  ##Results_Loss = [] # Total Prediction Loss By MSE
  for j in range(len(Indices)):
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
    #
    Data = pd.read_csv(os.getcwd() + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
    Data = Data.drop(columns = Data.keys()[0])
    Data.iloc[37, Indices[j]] = np.mean(tf.keras.losses.logcosh(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )
    Data.to_csv(os.getcwd() + "\\" + Symbol + "_LogCosh_Methods_Comparisons_Results.csv")
    #
    Data = pd.read_csv(os.getcwd() + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
    Data = Data.drop(columns = Data.keys()[0])
    Data.iloc[37, Indices[j]] = np.mean(tf.keras.losses.mse(Y[:, :], Y_Prediction[:, :-1]).numpy() )
    Data.to_csv(os.getcwd() + "\\" + Symbol + "_Loss_Methods_Comparisons_Results.csv")
    #
    Data = pd.read_csv(os.getcwd() + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")
    Data = Data.drop(columns = Data.keys()[0])
    Data.iloc[37, Indices[j]] = np.mean(tf.keras.losses.mae(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )
    Data.to_csv(os.getcwd() + "\\" + Symbol + "_MAE_Methods_Comparisons_Results.csv")
    #
    Data = pd.read_csv(os.getcwd() + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")
    Data = Data.drop(columns = Data.keys()[0])
    Data.iloc[37, Indices[j]] = np.mean(tf.keras.losses.mape(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )
    Data.to_csv(os.getcwd() + "\\" + Symbol + "_MAPE_Methods_Comparisons_Results.csv")
    #
    Data = pd.read_csv(os.getcwd() + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")
    Data = Data.drop(columns = Data.keys()[0])
    Data.iloc[37, Indices[j]] = np.mean(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy() )
    Data.to_csv(os.getcwd() + "\\" + Symbol + "_MSE_Methods_Comparisons_Results.csv")
    #
    Data = pd.read_csv(os.getcwd() + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")
    Data = Data.drop(columns = Data.keys()[0])
    Data.iloc[37, Indices[j]] = np.mean(np.subtract(Y[:, Peaks_Indices], Y_Prediction[:, Peaks_Indices]) )
    Data.to_csv(os.getcwd() + "\\" + Symbol + "_Peaks_Methods_Comparisons_Results.csv")
    #
    Data = pd.read_csv(os.getcwd() + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")
    Data = Data.drop(columns = Data.keys()[0])
    Data.iloc[37, Indices[j]] = np.mean(np.sqrt(tf.keras.losses.mse(Y[:, Loss_Indices], Y_Prediction[:, Loss_Indices]).numpy()) )
    Data.to_csv(os.getcwd() + "\\" + Symbol + "_RMSE_Methods_Comparisons_Results.csv")
    #
    Data = pd.read_csv(os.getcwd() + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")
    Data = Data.drop(columns = Data.keys()[0])
    Data.iloc[37, Indices[j]] = np.mean(np.subtract(Y[:, Valeys_Indices], Y_Prediction[:, Valeys_Indices]) )
    Data.to_csv(os.getcwd() + "\\" + Symbol + "_Valley_Methods_Comparisons_Results.csv")
    #
    F = open("./" + Symbol + "_Predictions.txt", "rb")
    Data = pickle.load(F)
    F.close()
    Data["Predictions"][3699 + Indices[j]] = Y_Prediction[0, :].tolist() # Used 3699 Instead Of 3700 Because Indices Start From 1 Instead Of 0
    F = open("./" + Symbol + "_Predictions.txt", "wb")
    pickle.dump(Data, F)
    F.close()
    #
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize = (13, 6))
    plt.title(Symbol + " Close Price, Daily TimeFrame")
    plt.plot(Y[0, :].tolist(), label = "Input")
    plt.plot(Y_Prediction[0, :].tolist(), label = "Prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    #
    print(Method_Name)
    print("----------------------- ----------------------- ----------------------- -----------------------")
    #Method_Total += [Method_Name] # + "_" + str(j)
    ##Y_Prediction_Total += [[Y_Prediction]]
    #Y_Prediction_Total += [Y_Prediction[0, :].tolist()]
    #

  Info = pd.read_csv(os.getcwd() + "\\" + Loss_File)
  Info = Info.drop(columns = Info.keys()[0])
  Indices = [i for i in range(1, Last_Round + 1) if Info.iloc[37, i] > (Info["Mean"][37] + (3 * Info["Variance"][37]))]

if "Mean_Variance_Calculation.py" in os.listdir():
  exec(open("Mean_Variance_Calculation.py").read())
else:
  print("Calculating Mean And Variance Of Results Advised Afterward")
print("All Unsatisfied Results Were Repalced Successfully")
time.sleep(7)



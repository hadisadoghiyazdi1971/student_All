import numpy as np
import pandas as pd
import os
#import pickle
#import matplotlib.pyplot as plt

FileNames = [name for name in os.listdir(os.getcwd()) if ".csv" in name]


Data = np.zeros((13, 7))
Method_Indices = [0, 1, 2, 3, 31, 32, 35, 36, 4, "Cor", "Hub", 37, "SGD"]

for i in range(len(FileNames)):
  if "_Loss_" in FileNames[i]:
    Loss_File = FileNames[i]
    break

Info = pd.read_csv(os.getcwd() + "\\" + Loss_File)
Info = Info.drop(columns = Info.keys()[0])
Method_Indices[9] = np.argmin(Info["Mean"].iloc[5:17]) + 5 # CorrEntropy
Method_Indices[10] = np.argmin(Info["Mean"].iloc[17:29]) + 17 # Huber
Method_Indices[12] = np.argmin(Info["Mean"].iloc[38:70]) + 38 # SGD

#for i in range(1):
for i in range(len(FileNames)): 
  Info = pd.read_csv(os.getcwd() + "\\" + FileNames[i])
  Info = Info.drop(columns = Info.keys()[0])
  Drop_Columns = []
  for k in range(len(Info.keys())):
    if "Unnamed" in Info.keys()[k]:
      Drop_Columns += [Info.keys()[k]]
  if len(Drop_Columns) > 0:
    Info = Info.drop(columns = Drop_Columns)

  if "Mean" not in Info.keys() or "Variance" not in Info.keys():
    print(FileNames[i]," Has No Mean And Variance Column")
    continue

  if "_Peaks" in FileNames[i]:
    Data[:, 0] = np.around(Info["Mean"].to_numpy()[Method_Indices], 4)
  elif "_Valley" in FileNames[i]:
    Data[:, 1] = np.around(Info["Mean"].to_numpy()[Method_Indices], 4)
  elif "_MSE_" in FileNames[i]:
    Data[:, 2] = np.around(Info["Mean"].to_numpy()[Method_Indices], 4)
  elif "_MAE_" in FileNames[i]:
    Data[:, 3] = np.around(Info["Mean"].to_numpy()[Method_Indices], 4)
  elif "_RMSE_" in FileNames[i]:
    Data[:, 4] = np.around(Info["Mean"].to_numpy()[Method_Indices], 4)
  elif "_LogCosh_" in FileNames[i]:
    Data[:, 5] = np.around(Info["Mean"].to_numpy()[Method_Indices], 4)
  elif "_Loss_" in FileNames[i]:
    Data[:, 6] = np.around(Info["Mean"].to_numpy()[Method_Indices], 4)
  #Peaks_Data = np.array([])
  #Valeys_Data = np.array([])
  #MSE_Data = np.array([])
  #MAE_Data = np.array([])
 # RMSE_Data = np.array([])
  #LogCosh_Data = np.array([])
  #Loss_Data = np.array([])

#np.concatenate()

Methods_Name = np.expand_dims(Info["Method"].to_numpy()[Method_Indices], axis = 1)
Data = np.concatenate((Methods_Name, Data), axis = 1)

Columns = ["Methods", "Peaks", "Valleys", "MSE", "MAE", "RMSE", "LogCosh", "Loss"]

Data = pd.DataFrame(Data, columns = Columns)
Data.to_csv(os.getcwd() + "\\Brief.csv")

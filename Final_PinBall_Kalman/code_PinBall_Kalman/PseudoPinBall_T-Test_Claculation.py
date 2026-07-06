import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind

FileNames = [name for name in os.listdir(os.getcwd()) if (".csv" in name and "Results" in name)]
FileNames.sort()

Test_Index = 37
Method_Indices = [0, 1, 2, 3, 31, 32, 35, 36, 4, "Cor", "Hub", "SGD"]

for i in range(len(FileNames)):
  if "_Loss_" in FileNames[i]:
    Loss_File = FileNames[i]
    break

Info = pd.read_csv(os.getcwd() + "\\" + Loss_File)
Info = Info.drop(columns = Info.keys()[0])
if "Mean" not in Info.keys():
  print("Mean Of Results Needed, Run Mean_Variance_Calculation.py")
  quit()
Method_Indices[9] = np.argmin(Info["Mean"].iloc[5:17]) + 5 # CorrEntropy
Method_Indices[10] = np.argmin(Info["Mean"].iloc[17:29]) + 17 # Huber
Method_Indices[11] = np.argmin(Info["Mean"].iloc[38:70]) + 38 # SGD
for j in range(len(Info.keys())):
  #for j in range(len(Info.keys()) - 1, -1 , -1):
    if "Round" in Info.keys()[-j]:
      break

Last_Round = len(Info.keys()) - j

Columns = Info["Method"][Method_Indices].tolist()

Methods = []
Results = []
#for i in range(1):
for i in range(len(FileNames)):
  Methods += [FileNames[i][FileNames[i].find("_") + 1:FileNames[i].find("_Me")]]
  Info = pd.read_csv(os.getcwd() + "\\" + FileNames[i])
  Info = Info.drop(columns = Info.keys()[0])

  Temp_Results = []
  for j in range(len(Method_Indices)):
    Temp_Results += [ttest_ind(Info.iloc[Test_Index, 1:Last_Round], Info.iloc[Method_Indices[j], 1:Last_Round], equal_var = True)[1]]
  
  Results += [Temp_Results]

Results = np.around(np.array(Results), 4)
Methods = np.expand_dims(np.array(Methods), axis = 1)
Columns = ["Measurement"] + Columns

Data = np.concatenate((Methods, Results), axis = 1)

Data = pd.DataFrame(Data, columns = Columns)
Data.to_csv("./" + FileNames[0][:FileNames[0].find("_")] +"_PseudoPinBall_T-Test.csv", index = False)

"""
Data = pd.read_csv("Temp.csv")

T_Test = ttest_ind(Data.iloc[0, :-1], Data.iloc[1, :-1], equal_var = True)
P_Value = T_Test[1]
T_Test = T_Test[0]
"""

# Second Method, By Numpy
# Difference = np.subtract(Data.iloc[0, :-1], Data.iloc[1, :-1])
# T_Value = np.mean(Difference) / (np.std(Difference, ddof = 1) / np.sqrt(len(Difference)))
# S_Value = np.random.standard_t(len(Difference), size = 100000)
# P_Value = np.sum(S_Value < T_Value) / float(len(S_Value))
# P_Value = 2 * min(P_Value, 1 - P_Value)

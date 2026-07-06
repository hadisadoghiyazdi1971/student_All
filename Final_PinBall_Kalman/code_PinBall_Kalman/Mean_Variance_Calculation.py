import numpy as np
import pandas as pd
import os

FileNames = [name for name in os.listdir(os.getcwd()) if ".csv" in name]
#FileNames = [name for name in os.listdir(os.getcwd()) if (".csv" in name and "Results" in name)]
#print(FileNames)

#FileNames = ["Synthesis_Valley_Methods_Comparisons_Results.csv"]


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
  
  if "Mean" in Info.keys() or "Variance" in Info.keys():
    continue
  Mean = np.array([])
  Variance = np.array([])

  Temp_Mean = []
  Temp_Variance = []
  for j in range(Info.shape[0]):
    #Mean = np.append(Mean, np.mean(Info.iloc[j, 1:]) )
    #Variance = np.append(Variance, np.var(Info.iloc[j, 1:]) )
    Temp = []
    for k in range(1, Info.shape[1]):
        if Info.iloc[j, k] < 31: Temp += [Info.iloc[j, k]]

    if len(Temp) != 0:
      Mean = np.append(Mean, np.mean(Temp) )
      Variance = np.append(Variance, np.var(Temp) )
    else:
      Mean = np.append(Mean, np.mean(Info.iloc[j, 1:]) )
      Variance = np.append(Variance, np.var(Info.iloc[j, 1:]) )
    
  Mean = np.expand_dims(Mean, axis =1)
  Variance = np.expand_dims(Variance, axis =1)
  Temp = np.concatenate((Mean, Variance), axis = 1)
  Columns = list(Info.keys())
  Columns += ["Mean", "Variance"]
  Info = Info.to_numpy()
  Info = np.concatenate((Info, Temp), axis = 1)
  Info = pd.DataFrame(Info, columns = Columns)

  Info.to_csv(os.getcwd() + "\\" + FileNames[i])

print()

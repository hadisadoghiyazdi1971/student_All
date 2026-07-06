import numpy as np
import matplotlib.pyplot as plt

def CorrEntropy(X, Sigma = 1):
  return np.subtract(1, np.exp(-(X ** 2)/(Sigma ** 2)))

def PseudoHuber(X, Delta = 1):
  return np.multiply((Delta ** 2), np.subtract(np.sqrt(np.add(1, (X/Delta) ** 2)), 1))

def PinBall(x, Tau = 0.5): 
  return np.add(np.multiply(np.array(x>=0).astype(np.int8), np.multiply(x, Tau)), np.multiply(np.array(x<0).astype(np.int8), np.multiply(x, (Tau - 1))))

X = np.arange(-499, 500) * 0.01

plt.figure(figsize = (17, 8))
plt.rcParams.update({"font.size": 24, "font.family": "Times New Roman"})
plt.title("Different Losses")
plt.plot(np.arange(-229, 230) * 0.01, np.power(np.arange(-229, 230) * 0.01, 2), label = r"MSE", color = "orange")
plt.plot(X, np.abs(X), label = r"MAE", color = "cyan")
plt.plot(X, CorrEntropy(X, 1), label = r"CorrEntropy ($\sigma = 1$)", color = "purple")
plt.plot(X, PseudoHuber(X, 1), label = r"PseudoHuber ($\delta = 1$)", color = "green")
plt.plot(X, PinBall(X, 0.7), label = r"PseudoPinBall ($\tau = 0.7$)", color = "red")
plt.xlabel("Error", fontsize = 19)
plt.ylabel("Loss", fontsize = 19)
plt.legend(fontsize = 19)
plt.grid()
plt.tight_layout()
plt.gca().set_xlim([-6.9, 6.9])
plt.savefig("./Taw_Explanataion_3.png")
plt.show()

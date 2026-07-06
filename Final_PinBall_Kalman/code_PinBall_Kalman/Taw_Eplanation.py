import numpy as np
import matplotlib.pyplot as plt

Y = np.tile(np.abs(np.arange(-23, 23)), 5)
Y = Y[:70]

X = list(range(len(Y)))
X_2 = np.subtract(X , 1)


plt.rcParams.update({"font.size": 24, "font.family": "Times New Roman"})
##plt.figure(figsize = (19, 4))
plt.figure(figsize = (12, 7))
##plt.subplot(1, 3, 1)

plt.gca().set_xlim([-3, 76])
plt.plot(X, Y, color = "blue", linewidth = 2.3, label = "Time Series (y)")
plt.plot(X_2, Y, color = "red", linewidth = 2.3, label = r"Prediction ($\hat{y}^*$)", linestyle = "--")
plt.legend(fontsize = 19, loc = 1)
##plt.grid()
plt.xlabel("Time Value (Day) Of Synthesis Stock")
plt.ylabel("Closing Price Of Synthesis Stock")

X_Scatter = [8, 8, 9, 38, 39, 52, 23, 46, 69]
Y_Scatter = [15, 14, 14, 15, 17, 16, 0, 23, 0]
plt.scatter(X_Scatter, Y_Scatter, c = "#000000")

Label_X = [8, 5, 10, 38.3, 32.5, 44.6, 23.5, 46.69, 69.46]
Label_Y = [16, 13, 14, 14, 17, 16, -0.89, 22.61, -0.89]
Label_Scatter = [r"$y_t$", r"$\hat{y}_t$", r"$y_{t+1}$", r"$d=0$", r"$d<0$", r"$d>0$"
                              , r"$t=a$", r"$t=b$", r"$t=c$"]
for i in range(len(Label_X)):
  plt.text(Label_X[i], Label_Y[i], Label_Scatter[i])

plt.vlines(8, 0, 15, linestyle = "--", color = "black")
plt.vlines(9, 0, 14, linestyle = "--", color = "black")
plt.hlines(14, 8, 9, linestyle = "--", color = "black")

plt.tight_layout()
plt.savefig("./Taw_Explanation_1_TimeSeries.png")

#####plt.show()
plt.figure(figsize = (12, 7))
##plt.subplot(1, 3, 2)

X = np.arange(-499, 500) * 0.1
Y = np.tanh(0.046 * X) / 2.5
Y= np.add(Y, 0.5)
plt.gca().set_ylim([0, 1])
plt.gca().set_xlim([-57, 57])
plt.plot(X, Y, color = "green", linewidth = 2.3, label = "Taw + Prior Knowledge")
#plt.legend(fontsize = 11, loc = 4)
##plt.grid()
plt.xlabel("d", style = "italic")
plt.ylabel(r"$\tau$") # "Taw"

#plt.hlines(0, -51, 53, "black")
#plt.vlines(0, 0, 1, "black")

plt.text(30.23, 0.80, "Downward Trend")
plt.text(-55, 0.15, "Upward Trend")
plt.scatter(0, 0.5, c= "#000000")
plt.hlines(0.11, -57, -49.7, linestyle = "--", color = "black")
plt.hlines(0.9, -57, 50, linestyle = "--", color = "black")

plt.scatter([-50, 50], [0.11, 0.89], c = "#000000")
plt.text(-50, 0.05, r"$b'$")
plt.text(47, 0.92, r"$a'$, $c'$")

plt.tight_layout()
plt.savefig("./Taw_Explanation_2_PriorTaw.png")
#####plt.show()


def CorrEntropy(X, Sigma = 1):
  return np.subtract(1, np.exp(-(X ** 2)/(Sigma ** 2)))

def PseudoHuber(X, Delta = 1):
  return np.multiply((Delta ** 2), np.subtract(np.sqrt(np.add(1, (X/Delta) ** 2)), 1))

def PinBall(x, Tau = 0.5): 
  return np.add(np.multiply(np.array(x>=0).astype(np.int8), np.multiply(x, Tau)), np.multiply(np.array(x<0).astype(np.int8), np.multiply(x, (Tau - 1))))


plt.figure(figsize = (12, 7))
plt.rcParams.update({"font.size": 24, "font.family": "Times New Roman"})
##plt.subplot(1, 3, 3)

X = np.arange(-499, 500) * 0.01
X_MSE = np.arange(-229, 230) * 0.01
Y_MSE = np.abs(np.power(X_MSE , 2))
Y_MAE = np.abs(X)
#Y_PseudoHuber = np.multiply((1 ** 2), np.subtract(np.sqrt(np.add(1, (X / 1.0) ** 2)), 1))
Y_PseudoHuber = PseudoHuber(X, 1)
#Y_CorrEntropy = np.subtract(1, np.exp( np.divide( (-1 * np.abs(np.power(X , 2))), 2 * (1 ** 2))))
Y_CorrEntropy = CorrEntropy(X, 1)
#Y_PinBall = np.add(np.multiply(np.array(X >= 0).astype(np.int8), np.multiply(X, 0.75)), np.multiply(np.array(X < 0).astype(np.int8), np.multiply(X, (0.75 - 1))))
Y_PinBall = PinBall(X, 0.75)


plt.gca().set_xlim([-7, 7])
plt.gca().set_ylim([-0.5, 6])

plt.plot(X_MSE, Y_MSE, label = r"MSE", color = "orange", linewidth = 3.1)
plt.plot(X, Y_MAE, label = r"MAE", color = "cyan", linewidth = 3.1)
plt.plot(X, Y_PseudoHuber, label = r"PseudoHuber ($\delta = 1$)", color = "green", linewidth = 3.1)
plt.plot(X, Y_CorrEntropy, label = r"CorrEntropy ($\sigma = 1$)", color = "purple", linewidth = 3.1)
plt.plot(X, Y_PinBall, label = r"PseudoPinBall ($\tau = 0.75$)", color = "red", linewidth = 3.1)

#plt.legend(fontsize = 19, loc = 9)
plt.legend(fontsize = 19, loc = 1)
plt.grid()
plt.xlabel("Error", fontsize = 19)
plt.ylabel("Loss", fontsize = 19)

plt.tight_layout()
plt.savefig("./Taw_Explanation_3_Losess.png")
plt.show()
##plt.savefig("./Taw_Explanation.png")

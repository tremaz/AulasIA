import numpy as np
#import plotly.express as px
import matplotlib.pyplot as plt

xdata = np.array([1,2,3])
ydata = np.array([1,4,8])
print(xdata)
print(ydata)

xbar = np.mean(xdata)
ybar = np.mean(ydata)

b1 = np.sum((ydata - ybar)* (xdata - xbar)) / np.sum((xdata -xbar)**2)
b0 = ybar - b1*xbar

y_ajustado = b0 + b1*xdata
r_score = 1 - ((np.sum(ydata - y_ajustado)**2) / np.sum((ydata - ybar)**2))

sqr = np.sum((ydata -y_ajustado)**2)
sqt = np.sum((ydata -ybar)**2)
r_score = 1 - (sqr/sqt)

print(r_score)

#fig = px.scatter(x=xdata, y=ydata,
#                 trendline="ols",
#                 labels={"x":"x", "y":"y"})

#fig.show()

plt.plot(xdata, ydata)
plt.xlabel("X")
plt.ylabel("")
plt.title("Visualização dos dados")
plt.show()
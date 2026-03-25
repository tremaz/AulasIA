import numpy as np
import plotly.express as px
class LinearRegression: #criando a classe
    def __init__(self,x, y): #construtor
        self.x = np.array(x) #dados de entrada
        self.y = np.array(y) #dados de saida
        self.b0 = None #intercepto
        self.b1 = None #coeficiente angular
    def fit(self): #treinamento, descobrir os parametros(b0, b1)
        xbar = np.mean(self.x)
        ybar = np.mean(self.y)
        self.b1 = np.sum((self.x - xbar) * (self.y - ybar))/ \
        np.sum((self.x - xbar) **2) #calculo do coeficiente angular
        self.b0 = ybar - self.b1 * xbar #calculo do intercepto
        return self
    def predict(self, x_new): #predicao
            return self.b0 + self.b1 * np.array(x_new)
    def summary(self):
        """ print(f"Modelo: y = {self.b0} + {self.b1} * x")
        print(f"Intercepto = {self.b0}")
        print(f"Coeficiente angular {self.b1}") """

dados = np.genfromtxt(r".\qb_2004.csv", dtype=float, delimiter=",", skip_header=1)[::-1]

X = dados[:,2]
Y = dados[:,3]
""" print("Dados de X", X)
print("Dados de Y", Y) """
""" fig = px.scatter(x=X, y=Y, title="Dados brutos")
fig.show() """

modelo = LinearRegression(X, Y)
modelo.fit()

print("Questão a)")
print("Inclinação: ", modelo.b1)
print("Intercepto: ", modelo.b0)
print(modelo.summary())
teste = [7.5]
print("Questão b)")
print("Pontuação media para 7,5 jardas: ", modelo.predict(teste))

teste2 = [6.5]
print("Questão c)")
print("A diferença que uma jarda faz por tentativa: ", (modelo.predict(teste) - modelo.predict(teste2)))

teste3 = [7.21]
print("Questão d)")
print("O valor observado para 7.21 jardas é de [95.0], já o ajustado é de: ", modelo.predict(teste3))
print("O residuo é de ", 95.0 - (modelo.predict(teste3)))

print("Questão e)")
valoresOB = []
valoresAJ = []
residuos = []

print("Observação i || Valor observado yi || Valor ajustado y^ || Residuo e")
for i in range(len(X)): 
     
     AuxOb = Y[i].item()
     AuxAj = modelo.predict(X[i]).item()
     res = (AuxOb - AuxAj)
     valoresOB.append(AuxOb)
     valoresAJ.append(AuxAj)
     residuos.append(res)
     print("       ",i+1,"    ||     ", AuxOb,"      ||    ", AuxAj," || ", res)



""" fig = px.scatter(x=X, y=Y, trendline="ols", labels={"x":"x","y":"y"})
fig.show() """


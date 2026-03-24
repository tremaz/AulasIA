import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class RegressionM:
  def __init__(self,X, y): # construtor
    self.X = X # entradas
    self.y = y # saida
    self.N = X.shape[0] # número de amostras
    self.beta = None

  def fit(self): #Treinamento (descoberta dos parametros)
    X_b = np.column_stack((np.ones((self.N, 1)), self.X))
    self.beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ self.y # Equação da regressão linear múltipla

  def predict(self, X_new): # preditivo, prever os valores de X_new
    if self.beta is None:
      raise ValueError("Modelo não treinado. Chame fit() antes de predict().")
    N = X_new.shape[0]
    X_new = np.column_stack((np.ones((N, 1)), X_new))
    return X_new @ self.beta

# Aplicação do algorítmo de regressão linear múltipla

script_dir = os.path.dirname(os.path.abspath(__file__))
dados = np.loadtxt(os.path.join(script_dir, 'dose_radiacao_expandido.csv'), delimiter=',', skiprows=1)
# print(dados)

dose_radiacao = dados[:, 1]  # variável alvo (dependente)
corrente = dados[:, 2]  # variável independente 1
tempo = dados[:, 3]  # variável independente 2

entradas = np.column_stack((corrente, tempo))  # junta corrente e tempo como features

modelo = RegressionM(entradas, dose_radiacao)
modelo.fit()
# print("Valores estimados dos parametros B0, B1 e B2\n", modelo.beta)
dose_radiacao_predita = modelo.predict(entradas)
# print(dose_radiacao_predita)

X_novo = np.array([[15, 5]]) # item b

print(f"Intercepto (b0): {modelo.beta[0]}")  # pyright: ignore[reportOptionalSubscript]
print(f"Coeficiente da corrente (b1): {modelo.beta[1]}")  # pyright: ignore[reportOptionalSubscript]
print(f"Coeficiente do tempo (b2): {modelo.beta[2]}")  # pyright: ignore[reportOptionalSubscript]

previsao = modelo.predict(X_novo) # item b

print(f"\nPrevisão para 15 mAmp e 5 min: {previsao[0]:.4f} rad") # item b

# Resíduos
residuos = dose_radiacao - dose_radiacao_predita  

print(f"{'Obs (i)':<5} | {'Real (yi)':<12} | {'Ajustado (y^i)':<18} | {'Resíduo (ei)':<12}")

for i in range(len(dose_radiacao)):
  print(f"{i+1:<5} | {dose_radiacao[i]:<12.2f} | {dose_radiacao_predita[i]:<18.2f} | {residuos[i]:<12.2f}")

media_residuo = np.mean(np.abs(residuos))
print(f"\nErro Médio Absoluto (MAE): {media_residuo:.2f} rad")

# Projeção do hiperplano
corrente_grid, tempo_grid = np.meshgrid(
  np.linspace(min(corrente), max(corrente), 10),
  np.linspace(min(tempo), max(tempo), 10)
)

def r2_score(y_true, y_pred):  # função genérica que calcula o r2_score
  numerador = np.sum((y_true - y_pred)**2)
  denominador = np.sum((y_true - np.mean(y_true))**2)
  r2 = 1 - (numerador/denominador)
  return r2

r2 = r2_score(dose_radiacao, dose_radiacao_predita)
print(f"R² = {r2}")

coeficientes = modelo.beta
assert coeficientes is not None, "Modelo não treinado."
dose_radiacao_grid = coeficientes[0] + coeficientes[1]*corrente_grid + coeficientes[2]*tempo_grid

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(xs=corrente, ys=tempo, zs=dose_radiacao, color="red")  # pyright: ignore[reportArgumentType]
ax.plot_surface(corrente_grid, tempo_grid, dose_radiacao_grid, alpha = 0.5, color = "blue")
ax.set_xlabel("Corrente")
ax.set_ylabel("Tempo")
ax.set_zlabel("Dose")
ax.set_title("Modelo de Regressão Linear Múltipla")
plt.show()

# gráfico interativo

fig = go.Figure() # cria uma figura

# dados reais
fig.add_scatter3d(x = corrente, y = tempo, z = dose_radiacao, mode = "markers", marker = dict(color = "red", size = 5), name = "Dados")

# dados previstos
fig.add_scatter3d(x = corrente, y = tempo, z = dose_radiacao_predita, mode = "markers", marker = dict(color = "green", size = 5), name = "Previstos")

# plano da regressão
fig.add_surface(x = corrente_grid, y = tempo_grid, z = dose_radiacao_grid, opacity = 0.5, colorscale = "blues", name = "Plano")

fig.show()

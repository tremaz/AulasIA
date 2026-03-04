import numpy as np

v = np.array([1, 2, 3]) #Letras minusculas, unidimencional
print(type(v))

A = np.array([[1,2, 9, 8],
              [1,5, 10, 2],
              [3,6,7,8],
              [6,7,9,5]]) #Letras maiusculas, bidimencional

M1 = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]]
)

M2 = np.array([[6,5,4],
              [3,2,1],
              [1,2,3]]
)

#print(M1 @ M2) #Mult de matrizes
#print(M1.T) #Matriz trasposta

print(np.empty(5)) #Cria com base no lixo da memoria
print(np.zeros(((2,5)))) #Cria com valores zeros

a = np.array([1,2,3])

print(np.max(a))
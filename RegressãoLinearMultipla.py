import numpy as np

x = np.array([[2, 50], [8, 110], [11, 120], [10, 550], [8, 295], [4, 200], [2, 375], [2, 52], [9, 100], [8, 300]])
xt = x.transpose()
    

P = x*xt

print(x)
print(xt)
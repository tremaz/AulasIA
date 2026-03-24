import numpy as np

data = np.genfromtxt(r".\qb_2004.csv", dtype=str, delimiter=",", skip_header=1, )
print(data)
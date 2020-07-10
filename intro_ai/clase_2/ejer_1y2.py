import numpy as np


X = np.array([
[1,2,3],
[4,5,6],
[7,8,9]
])
C = np.array([
[1,0,0],
[0,1,1]
])
expanded_C = C[:, None]
distances = np.sqrt(np.sum((expanded_C - X) ** 2, axis=2))
print("Distancias")
print(distances)
arg_min = np.argmin(distances, axis=0)
print("Distancias minimas")
print(arg_min)
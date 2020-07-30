import numpy as np

path = 'income.data.csv'
my_data = np.genfromtxt(path, delimiter=',', skip_header = 1)
print(my_data)



# Para agregar columna de 1 --> np.verticalstack
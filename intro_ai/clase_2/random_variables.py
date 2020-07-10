import numpy as np


# lambda_param=0.1
# u=np.random.uniform(0,1,100)
# x=-np.log(1-u)/lambda_param
#
#
# x = np.power(u,1/3)  # Ejercicio 6

def exponential_random_variable(lambda_param, size):
    uniform_random_variable = np.random.uniform(low=0.0, high=1.0, size=size)
    return (-1 / lambda_param) * np.log(1 - uniform_random_variable)

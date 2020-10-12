import numpy as np
import matplotlib.pyplot as plt


L = 500
x_iIz_i__0 = np.random.normal(5, 15, L)
x_iIz_i__1 = np.random.normal(5, 15, L)

x_i = np.random.uniform(0, 1, L)

mask_idx = 1*(x_i < 0.25)

idx_0 = np.where(mask_idx == 0)
idx_1 = np.where(mask_idx == 1)


x_gen = x_iIz_i__0[idx_0]
# x_gen = x_iIz_i__0[idx_0]

hist = np.histogram(x_iIz_i__0, 10)

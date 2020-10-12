#!/usr/bin/env python
# coding: utf-8

# In[118]:


#Si queremos que las imágenes sean mostradas en una ventana emergente quitar el inline
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib

# OpenCV-Python utiliza NumPy para el manejo de imágenes
import numpy as np
# cv2 es el módulo python para acceder a OpenCV 
import cv2 as cv
# Usamos las poderosas herramientas de graficación de matplotlib para mostrar imágenes, perfiles, histogramas, etc
import matplotlib.pyplot as plt


# In[119]:


def LBP_char(gray):
    gray = gray.astype('float64') 
    gray_LBP = np.zeros(gray.shape)

    for j in range(1,gray.shape[1]-1):
        for i in range(1,gray.shape[0]-1):
                x = ((gray[i,j]-gray[i,j-1])>=0)*2**0+((gray[i,j]-gray[i+1,j-1])>=0)*2**1+((gray[i,j]-gray[i+1,j])>=0)*2**2+((gray[i,j]-gray[i+1,j+1])>=0)*2**3+((gray[i,j]-gray[i,j+1])>=0)*2**4+((gray[i,j]-gray[i-1,j+1])>=0)*2**5+((gray[i,j]-gray[i-1,j])>=0)*2**6+((gray[i,j]-gray[i-1,j-1])>=0)*2**7
                gray_LBP[i,j] = x
#                 print(x)
    return np.uint8(np.around(gray_LBP[1:298,1:298]))
    


# In[120]:


# Extractor de carcacterísiticas LBP
# ==================================

# Piedras
# =======
img = cv.imread('TP4/piedras2.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('piedras1',gray)
cv.waitKey(0)
cv.destroyAllWindows()


gray_LBP = LBP_char(gray)
cv.imshow('piedras1',gray_LBP)
cv.waitKey(0)
cv.destroyAllWindows()

plt.hist(gray_LBP.ravel(),256,[0,256]); plt.show()


# In[121]:


# Ladrillos
# ==================================

img = cv.imread('TP4/ladrillos1.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('piedras1',gray)
cv.waitKey(0)
cv.destroyAllWindows()

gray_LBP = LBP_char(gray)
# print(gray_LBP)
cv.imshow('piedras1',gray_LBP)
cv.waitKey(0)
cv.destroyAllWindows()

plt.hist(gray_LBP.ravel(),256,[0,256]); plt.show()


# In[122]:


# Ladrillos
# ===========

img = cv.imread('TP4/cielo.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('piedras1',gray)
cv.waitKey(0)
cv.destroyAllWindows()

gray_LBP = LBP_char(gray)
# print(gray_LBP)
cv.imshow('piedras1',gray_LBP)
cv.waitKey(0)
cv.destroyAllWindows()

plt.hist(gray_LBP.ravel(),256,[0,256]); plt.show()


# In[123]:


# Oveja
# ===========

img = cv.imread('TP4/oveja2.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('piedras1',gray)
cv.waitKey(0)
cv.destroyAllWindows()

gray_LBP = LBP_char(gray)
# print(gray_LBP)
cv.imshow('piedras1',gray_LBP)
cv.waitKey(0)
cv.destroyAllWindows()

plt.hist(gray_LBP.ravel(),256,[0,256]); plt.show()


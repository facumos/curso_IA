# OpenCV-Python utiliza NumPy para el manejo de imágenes
import numpy as np
# cv2 es el módulo python para acceder a OpenCV
import cv2 as cv
# Usamos las poderosas herramientas de graficación de matplotlib para mostrar imágenes, perfiles, histogramas, etc
import matplotlib.pyplot as plt
# Importamos librerías para manejo de tiempo
import time


img = cv.imread('G:\My Drive\AI\CURSO\computer_vision\Im_TPs\TP2\metalgrid.jpg', 0)
# # grises = True
# grises = False
# if grises:
#     # En escala de grises
#     laplacian = cv.Laplacian(img, cv.CV_64F)
#     sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
#     sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
# else:
#     # En blanco y negro
#     laplacian = cv.Laplacian(img, cv.CV_8U)
#     sobelx = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=3)
#     sobely = cv.Sobel(img, cv.CV_8U, 0, 1, ksize=3)
#
# ax1 = plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# ax2 = plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
# plt.title('Laplaciano'), plt.xticks([]), plt.yticks([])
# ax3 = plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# ax4 = plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#
# plt.show()
#
#
# img = cv.imread('golazo.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # Aplico Canny
# # =============
# start = time.time()
# edges = cv.Canny(img, 40, 105, L2gradient=True)
# elapsed = time.time()-start
# print('Tiempo de procesamiento {} segundos'.format(elapsed))
#
# # Muestro la imagen
# # ==================
# cv.namedWindow("Canny", 0)
# # cv.imshow("Canny", edges)
# plt.imshow(edges, cmap='gray')
# plt.show()

blur = cv.GaussianBlur(img, (5, 5), 0)

# Aplicamos Sobelx y Sobely en 'float32', luego encontramos el módulo y el ángulo:
sobelx_64 = cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=3)
sobely_64 = cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=3)

mod_G = np.sqrt(sobelx_64**2 + sobely_64**2)
angle_G = np.arctan(np.divide(sobely_64, sobelx_64+0.00001))

abs_64 = np.absolute(mod_G)
mod_G_8u1 = abs_64/abs_64.max()*255
mod_G_8u = np.uint8(mod_G_8u1)

abs_64 = np.absolute(angle_G)
angle_G_8u1 = abs_64/abs_64.max()*255
angle_G_8u = np.uint8(angle_G_8u1)

ax1 = plt.subplot(3, 1, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
ax2 = plt.subplot(3, 1, 2), plt.imshow(mod_G_8u, cmap='gray')
plt.title('Modulo'), plt.xticks([]), plt.yticks([])
ax3 = plt.subplot(3, 1, 3), plt.imshow(angle_G_8u, cmap='gray')
plt.title('Angulo'), plt.xticks([]), plt.yticks([])
plt.show()

mod_max_0 = np.max(mod_G_8u, axis=0)
mod_max_1 = np.max(mod_G_8u, axis=1)
mod_max = np.array([mod_max_0, mod_max_1])
abs_64 = np.absolute(mod_max)
mod_max_8u1 = abs_64/abs_64.max()*255
mod_max_8u = np.uint8(mod_max_8u1)
ax1 = plt.subplot(2, 1, 1), plt.imshow(mod_G_8u, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
ax2 = plt.subplot(2, 1, 2), plt.imshow(mod_max_8u)
plt.show()
angle_max_0 = np.max(angle_G_8u, axis=0)
angle_max_1 = np.max(angle_G_8u, axis=1)




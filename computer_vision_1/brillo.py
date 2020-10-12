# Operadores de píxel
# Brillo Contraste Gamma

# Si queremos que las imágenes sean mostradas en una ventana emergente quitar el inline
# %matplotlib

# OpenCV-Python utiliza NumPy para el manejo de imágenes
import numpy as np
# cv2 es el módulo python para acceder a OpenCV
import cv2 as cv
# Usamos las poderosas herramientas de graficación de matplotlib para mostrar imágenes, perfiles, histogramas, etc
import matplotlib.pyplot as plt
# Using matplotlib backend: Qt5Agg

# Cargar una imagen en modo monocromático (un canal)
# img = cv.imread('imgBloque1.bmp',cv.IMREAD_GRAYSCALE)
img = cv.imread('G:\My Drive\AI\CURSO\computer_vision\Im_TPs\TP1\CoordCrom_1.png', cv.IMREAD_COLOR)
# img = cv.imread('G:\My Drive\AI\CURSO\computer_vision\Im_TPs\TP1\CoordCrom_2.png', cv.IMREAD_GRAYSCALE)

# imread lee la imagen en GBR, la pasamos a RGB:
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()

# Nueva figura
fig = plt.figure()

# Imagen original
ax1 = plt.subplot(221)
ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Original')

hist1, bins1 = np.histogram(img.ravel(), 256, [0, 256])

# Modificamos contraste 10%
contraste = 10
img_mod = (1+contraste/100)*img
img_mod = img_mod.round()
np.clip(img_mod, 0, 255, out=img_mod)   # Clip trunca a lo que se le diga (0 a 255)
img_mod = img_mod.astype('uint8')  # Hay que castear a 8 bits
ax2 = plt.subplot(222)
ax2.imshow(img_mod, cmap='gray', vmin=0, vmax=255)
ax2.set_title('Contraste: {}%'.format(contraste))

hist2, bins2 = np.histogram(img_mod.ravel(), 256, [0, 256])

# Modificamos contraste 50%
contraste = 50
img_mod = (1+contraste/100)*img
np.clip(img_mod, 0, 255, out=img_mod)   # Clip trunca a lo que se le diga (0 a 255)
img_mod = img_mod.astype('uint8')
ax3 = plt.subplot(223)
ax3.imshow(img_mod, cmap='gray', vmin=0, vmax=255)
ax3.set_title('Contraste: {}%'.format(contraste))

hist3, bins3 = np.histogram(img_mod.ravel(), 256, [0, 256])

# Modificamos contraste 80%
contraste = 80
img_mod = (1+contraste/100)*img
np.clip(img_mod, 0, 255, out=img_mod)   # Clip trunca a lo que se le diga (0 a 255)
img_mod = img_mod.astype('uint8')
ax4 = plt.subplot(224)
ax4.imshow(img_mod, cmap='gray', vmin=0, vmax=255)
ax4.set_title('Contraste: {}%'.format(contraste))

hist4, bins4 = np.histogram(img_mod.ravel(), 256, [0, 256])

plt.show()

# Nueva figura
fig = plt.figure()

# Histogramas de la imagen
plt.subplot(221), plt.plot(hist1)
plt.subplot(222), plt.plot(hist2)
plt.subplot(223), plt.plot(hist3)
plt.subplot(224), plt.plot(hist4)
plt.show()
# Ejemplo: Cambio de brillo
# In [6]:
# Nueva figura
fig = plt.figure()

# Imagen original
ax1 = plt.subplot(221)
ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Original')

# Modificamos brillo 10%
img_mod = img+(255*0.1)
np.clip(img_mod, 0, 255, out=img_mod)   # Clip trunca a lo que se le diga (0 a 255)
img_mod = img_mod.astype('uint8')         # Convierto a 8 bits
ax2 = plt.subplot(222)
ax2.imshow(img_mod, cmap='gray', vmin=0, vmax=255)
ax2.set_title('Brillo: 10%')

hist2, bins2 = np.histogram(img_mod.ravel(), 256, [0, 256])

# Modificamos brillo 50%
img_mod = img+(255*0.5)
np.clip(img_mod, 0, 255, out=img_mod)   # Clip trunca a lo que se le diga (0 a 255)
img_mod = img_mod.astype('uint8')         # Convierto a 8 bits
ax3 = plt.subplot(223)
ax3.imshow(img_mod, cmap='gray', vmin=0, vmax=255)
ax3.set_title('Brillo: 50%')

hist3, bins3 = np.histogram(img_mod.ravel(), 256, [0, 256])

# Modificamos brillo 80%
img_mod = img+(255*0.8)
np.clip(img_mod, 0, 255, out=img_mod)   # Clip trunca a lo que se le diga (0 a 255)
img_mod = img_mod.astype('uint8')         # Convierto a 8 bits
# print(np.amin(img_mod))
ax4 = plt.subplot(224)
ax4.imshow(img_mod, cmap='gray', vmin=0, vmax=255)
ax4.set_title('Brillo: 80%')

hist4, bins4 = np.histogram(img_mod.ravel(), 255, [0, 255])

plt.show()

# Nueva figura
fig = plt.figure()

# Histogramas de la imagen
plt.subplot(221), plt.plot(hist1)
plt.subplot(222), plt.plot(hist2)
plt.subplot(223), plt.plot(hist3)
plt.subplot(224), plt.plot(hist4)
plt.show()
# Corrección por Gamma
# In [7]:
gamma = 1.1
img_mod = np.power(img, gamma)
ax1 = plt.subplot(121)
ax1.set_title('Original')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
ax2 = plt.subplot(122)
ax2.set_title('Corregido por Gamma:1.1')
plt.imshow(img_mod, cmap='gray', vmin=0, vmax=255)

plt.show()

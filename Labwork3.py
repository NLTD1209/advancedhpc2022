from numba import cuda
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('000054.JPG')
img.save('sample.png')
 
numpy_img = np.asarray(img)

PixCount = numpy_img.shape[0]* numpy_img.shape[1]


numpy_img = numpy_img.reshape(PixCount,3)
print(numpy_img[500000])


def cpu_greyscale(img):
  new_img = np.zeros((img.shape[0]))
  for i in range(len(img)):
    pixel = 0
    for j in range(3):
      pixel += img[i][j]
    pixel = pixel /3
    new_img[i] = pixel
  return(new_img)
greyscale_img = cpu_greyscale(numpy_img)
print(greyscale_img[500000])

greyscale_img = greyscale_img.reshape(2642,3930)
output_img = Image.fromarray(greyscale_img)
plt.imshow(greyscale_img, cmap='gray', vmin=0, vmax=255)
plt.show()

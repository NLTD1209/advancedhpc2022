from numba import cuda
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import math

img = Image.open('000054.JPG')
img = img.resize((393,264))
img.save('sample.png')
 
numpy_img = np.asarray(img)
width, height = numpy_img.shape[0], numpy_img.shape[1]
PixCount = numpy_img.shape[0]* numpy_img.shape[1]


numpy_img = np.pad(numpy_img, [(3,3),(3,3),(0,0)])
print(numpy_img.shape)

filter = [[0, 0, 1, 2, 1, 0, 0],
[0, 3, 13, 22, 13, 3, 0],
[1, 13, 59, 97, 59, 13, 1],
[2, 22, 97, 159, 97, 22, 2],
[1, 13, 59, 97, 59, 13, 1],
[0, 3, 13, 22, 13, 3, 0],
[0, 0, 1, 2, 1, 0, 0]]

filter = np.array(filter)
print(np.sum(filter))

def gaussian_cpu(image):
  img = image.astype('float64')
  new_img = np.zeros((img.shape))
  for i in range(3,len(img)-3):
    for j in range(3,len(img[0])-3):
      sum = [0,0,0]
      for m in range(-3,4):
        for n in range(-3,4):
          sum = sum + img[i+m][j+n]*filter[m+3][n+3]
      sum = sum/1003
      new_img[i][j] = sum
  return(new_img)

blur_img = gaussian_cpu(numpy_img)
blur_img = blur_img.astype('uint8')
print(blur_img[200][200])

plt.imshow(blur_img)
plt.show()



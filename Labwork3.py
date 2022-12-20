from numba import cuda
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import math

img = Image.open('000054.JPG')
img.save('sample.png')
 
numpy_img = np.asarray(img)

PixCount = numpy_img.shape[0]* numpy_img.shape[1]


numpy_img = numpy_img.reshape(PixCount,3)


def cpu_greyscale(img):
  new_img = np.zeros((img.shape[0]))
  for i in range(len(img)):
    pixel = 0
    for j in range(3):
      pixel += img[i][j]
    pixel = pixel /3
    new_img[i] = pixel
  return(new_img)
time1 = time.time()
greyscale_img = cpu_greyscale(numpy_img)
print('CPU time in seconds: ', time.time()-time1)


greyscale_img = greyscale_img.reshape(2642,3930)
output_img = Image.fromarray(greyscale_img)
plt.imshow(greyscale_img, cmap='gray', vmin=0, vmax=255)
plt.show()



@cuda.jit
def GPU_grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = (src[tidx, 0] + src[tidx, 1] + src[tidx, 2])/3
    dst[tidx] = g

blocksize = 64
blocksize_range = [16,32,64,128,256,512,1024]
devOutput = cuda.device_array((PixCount), np.float64)
devInput = cuda.to_device(numpy_img)

for blocksize in blocksize_range:
  gridSize = math.ceil(PixCount / blocksize)
  time1 = time.time()
  GPU_grayscale[gridSize, blocksize](devInput, devOutput)
  print('GPU time in seconds of blocksize ',blocksize,' : ', time.time()-time1)

hostOutput = devOutput.copy_to_host()

greyscale_img = hostOutput.reshape(2642,3930)
plt.imshow(greyscale_img, cmap='gray', vmin=0, vmax=255)
plt.show()
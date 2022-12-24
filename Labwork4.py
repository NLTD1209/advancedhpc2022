from numba import cuda
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import math

img = Image.open('000054.JPG')
img.save('sample.png')
 
numpy_img = np.asarray(img)
width, height = numpy_img.shape[0], numpy_img.shape[1]
PixCount = numpy_img.shape[0]* numpy_img.shape[1]


numpy_img_1D = numpy_img.reshape(PixCount,3)


def cpu_greyscale_1D(img):
  new_img = np.zeros((img.shape[0]))
  for i in range(len(img)):
    pixel = 0
    for j in range(3):
      pixel += img[i][j]
    pixel = pixel /3
    new_img[i] = pixel
  return(new_img)
'''  
time1 = time.time()
greyscale_img = cpu_greyscale_1D(numpy_img_1D)
print('1D - CPU time in seconds: ', time.time()-time1)


greyscale_img = greyscale_img.reshape(2642,3930)
output_img = Image.fromarray(greyscale_img)
plt.imshow(greyscale_img, cmap='gray', vmin=0, vmax=255)
plt.show()
'''


@cuda.jit
def GPU_grayscale_1D(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = (src[tidx, 0] + src[tidx, 1] + src[tidx, 2])/3
    dst[tidx] = g


blocksize_range = [16,32,64,128,256,512,1024]


for blocksize in blocksize_range:
  devOutput = cuda.device_array((PixCount), np.float64)
  devInput = cuda.to_device(numpy_img_1D)
  gridSize = math.ceil(PixCount / blocksize)
  time1 = time.time()
  GPU_grayscale_1D[gridSize, blocksize](devInput, devOutput)
  hostOutput = devOutput.copy_to_host()
  print('1D - GPU time in seconds of blocksize ',blocksize,' : ', time.time()-time1)



greyscale_img = hostOutput.reshape(2642,3930)
plt.imshow(greyscale_img, cmap='gray', vmin=0, vmax=255)
plt.show()


def cpu_greyscale_2D(img):
  new_img = np.zeros((img.shape[0], img.shape[1]))
  for i in range(len(img)):
    for j in range(len(img[i])):
      pixel = 0
      for k in range(3):
        pixel += img[i][j][k]
      pixel = pixel /3
      new_img[i][j] = pixel
  return(new_img)
time1 = time.time()
greyscale_img = cpu_greyscale_2D(numpy_img)
print('2D - CPU time in seconds: ', time.time()-time1)


@cuda.jit
def GPU_grayscale_2D(src, dst):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
  g = (src[tidx, tidy, 0] + src[tidx, tidy, 1] + src[tidx, tidy, 2])/ 3
  dst[tidx, tidy] = g
  
blockSize_range = [(2,2),(4,4),(8,8),(16,16),(32,32)]
for blockSize in blockSize_range:
  gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))
  devOutput = cuda.device_array((width,height), np.float64)
  devInput = cuda.to_device(numpy_img)
  time1 = time.time()
  GPU_grayscale_2D[gridSize, blockSize](devInput, devOutput)
  hostOutput = devOutput.copy_to_host()
  print('2D - GPU time in seconds of blocksize ',blockSize,' : ', time.time()-time1)

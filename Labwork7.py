from numba import cuda, jit
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import math

img = Image.open('000054.JPG')
#img = img.resize((393,264))
img.save('sample.png')


 
numpy_img = np.asarray(img)
numpy_img = numpy_img.astype('float64')

width, height = numpy_img.shape[0], numpy_img.shape[1]
PixCount = numpy_img.shape[0]* numpy_img.shape[1]



@cuda.jit
def greyscale(src, dst):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    g = (src[i, j, 0] + src[i, j, 1] + src[i, j, 2]) / 3
    dst[i, j] = g
    
@cuda.reduce
def find_max(a, b):
    if a > b:
        return a
    else:
        return b

@cuda.reduce
def find_min(a, b):
    if a < b:
        return a
    else:
        return b

@cuda.jit
def stretch(src, dst, min_int, max_int):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    dst[tidx, tidy] = (src[tidx, tidy] - min_int) / (max_int - min_int) * 255




blockSize_range = [(2,2),(4,4),(8,8),(16,16),(32,32)]
for blockSize in blockSize_range:
  gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))
  devOutput1 = cuda.device_array((width, height), np.float64)
  devInput1 = cuda.to_device(numpy_img)
  time1 = time.time()

  greyscale[gridSize, blockSize](devInput1, devOutput1)
  grey_img = devOutput1.copy_to_host()

  devInput2 = cuda.to_device(grey_img.flatten())
  min_int = find_min(devInput2)
  max_int = find_max(devInput2)

  devOutput2 = cuda.device_array((width,height), np.float64)
  stretch[gridSize, blockSize](devOutput1, devOutput2, min_int, max_int)
  hostOutput = devOutput2.copy_to_host()
  print('GPU time in seconds of blocksize ',blockSize,' : ', time.time()-time1)


stretch_img = hostOutput.astype('uint8')
plt.imshow(stretch_img, cmap= 'gray')
plt.show()


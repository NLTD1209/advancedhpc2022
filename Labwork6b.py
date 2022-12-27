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



@jit
def mapfunc(val, th):
    new_val = val+th
    new_val = max(new_val,0)
    new_val = min(new_val,255)
    return new_val

@cuda.jit
def change_brit(src, dst, th):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y


    g = np.float64(mapfunc(np.uint8(src[tidx, tidy, 0]), th))
    dst[tidx,tidy,0] = g
    g = np.float64(mapfunc(np.uint8(src[tidx, tidy, 1]), th))
    dst[tidx,tidy,1] = g
    g = np.float64(mapfunc(np.uint8(src[tidx, tidy, 2]), th))
    dst[tidx,tidy,2] = g




blockSize_range = [(2,2),(4,4),(8,8),(16,16),(32,32)]
for blockSize in blockSize_range:
  gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))
  devOutput = cuda.device_array((numpy_img.shape), np.float64)
  devInput = cuda.to_device(numpy_img)
  time1 = time.time()
  change_brit[gridSize, blockSize](devInput, devOutput, -40)
  hostOutput = devOutput.copy_to_host()
  print('GPU time in seconds of blocksize ',blockSize,' : ', time.time()-time1)


binary_img = hostOutput.astype('uint8')
plt.imshow(binary_img)
plt.show()


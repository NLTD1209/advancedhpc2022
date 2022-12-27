from numba import cuda, jit
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import math

img = Image.open('000054.JPG')
#img = img.resize((393,264))
img.save('sample.png')

img2 = Image.open('DSCF8226.JPG')


 
numpy_img = np.asarray(img)
numpy_img = numpy_img.astype('float64')
numpy_img2 = np.asarray(img2)
numpy_img2 = numpy_img2.astype('float64')
width, height = numpy_img.shape[0], numpy_img.shape[1]
PixCount = numpy_img.shape[0]* numpy_img.shape[1]



@jit
def mapfunc(val1, val2, c):
    a = c*val1 + (1-c)*val2
    return a

@cuda.jit
def blend(src1, src2, dst, th):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y


    dst[tidx, tidy, 0] = mapfunc(
        src1[tidx, tidy, 0], src2[tidx, tidy, 0], th)
    dst[tidx, tidy, 1] = mapfunc(
        src1[tidx, tidy, 1], src2[tidx, tidy, 1], th)
    dst[tidx, tidy, 2] = mapfunc(
        src1[tidx, tidy, 2], src2[tidx, tidy, 2], th)




blockSize_range = [(2,2),(4,4),(8,8),(16,16),(32,32)]
for blockSize in blockSize_range:
  gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))
  devOutput = cuda.device_array((numpy_img.shape), np.float64)
  devInput1 = cuda.to_device(numpy_img)
  devInput2 = cuda.to_device(numpy_img2)
  time1 = time.time()
  blend[gridSize, blockSize](devInput1,devInput2, devOutput, 0.7)
  hostOutput = devOutput.copy_to_host()
  print('GPU time in seconds of blocksize ',blockSize,' : ', time.time()-time1)


blend_img = hostOutput.astype('uint8')
plt.imshow(blend_img)
plt.show()


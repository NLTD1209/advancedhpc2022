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
numpy_img = numpy_img/255

width, height = numpy_img.shape[0], numpy_img.shape[1]
PixCount = numpy_img.shape[0]* numpy_img.shape[1]



@cuda.jit
def RGB2HSV(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    max_ch = max(src[tidx, tidy, 0], src[tidx, tidy, 1], src[tidx, tidy, 2])
    min_ch = min(src[tidx, tidy, 0], src[tidx, tidy, 1], src[tidx, tidy, 2])
    delta = max_ch - min_ch

    if delta == 0:
        dst[0, tidx, tidy] = 0
    elif max_ch == src[tidx, tidy, 0]:
        dst[0, tidx, tidy] = 60 * (((src[tidx, tidy, 1]-src[tidx, tidy, 2])/delta) % 6)
    elif max_ch == src[tidx, tidy, 1]:
        dst[0, tidx, tidy] = 60 * (((src[tidx, tidy, 2]-src[tidx, tidy, 0])/delta)+2)
    elif max_ch == src[tidx, tidy, 2]:
        dst[0, tidx, tidy] = 60 * (((src[tidx, tidy, 0]-src[tidx, tidy, 1])/delta)+4)

    if max_ch == 0:
        dst[1, tidx, tidy] = 0
    else:
        dst[1, tidx, tidy] = delta/max_ch

    dst[2, tidx, tidy] = max_ch


@cuda.jit
def HSV2RGB(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    d = src[0, tidx, tidy]/60
    hi = int(d) % 6
    f = d - hi

    l = src[2, tidx, tidy]*(1-src[1, tidx, tidy])
    m = src[2, tidx, tidy]*(1-f*src[1, tidx, tidy])
    n = src[2, tidx, tidy]*(1-(1-f)*src[1, tidx, tidy])

    if 0 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 60:
        dst[tidx, tidy, 0] = src[2, tidx, tidy]
        dst[tidx, tidy, 1] = n
        dst[tidx, tidy, 2] = l
    elif 60 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 120:
        dst[tidx, tidy, 0] = m
        dst[tidx, tidy, 1] = src[2, tidx, tidy]
        dst[tidx, tidy, 2] = l
    elif 120 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 180:
        dst[tidx, tidy, 0] = l
        dst[tidx, tidy, 1] = src[2, tidx, tidy]
        dst[tidx, tidy, 2] = n
    elif 180 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 240:
        dst[tidx, tidy, 0] = l
        dst[tidx, tidy, 1] = m
        dst[tidx, tidy, 2] = src[2, tidx, tidy]
    elif 240 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 300:
        dst[tidx, tidy, 0] = n
        dst[tidx, tidy, 1] = l
        dst[tidx, tidy, 2] = src[2, tidx, tidy]
    elif 300 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 360:
        dst[tidx, tidy, 0] = src[2, tidx, tidy]
        dst[tidx, tidy, 1] = l
        dst[tidx, tidy, 2] = m



blockSize_range = [(2,2),(4,4),(8,8),(16,16),(32,32)] #
for blockSize in blockSize_range:
  gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))
  HSVOutput = cuda.device_array((3, width, height), np.float64)
  devInput = cuda.to_device(numpy_img)
  time1 = time.time()

  RGB2HSV[gridSize, blockSize](devInput, HSVOutput)
  hostOutput1 = HSVOutput.copy_to_host()
  RGBOutput = cuda.device_array((numpy_img.shape), np.float64)
  HSV2RGB[gridSize, blockSize](HSVOutput, RGBOutput)
  hostOutput = RGBOutput.copy_to_host()
  print('GPU time in seconds of blocksize ',blockSize,' : ', time.time()-time1)


Ori_img = numpy_img
plt.imshow(Ori_img)
plt.show()

RGB_img = hostOutput
plt.imshow(RGB_img)
plt.show()

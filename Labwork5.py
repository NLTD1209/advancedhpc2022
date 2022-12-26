from numba import cuda
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

def gaussian_cpu(img):
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

#time1 = time.time()
#blur_img = gaussian_cpu(numpy_img)
#print('cpu time: ', time.time()- time1)
#blur_img = blur_img.astype('uint8')
#print(blur_img[200][200])

#plt.imshow(blur_img)
#plt.show()




@cuda.jit
def gaussian_gpu(src, dst, filter):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y


    sumR = 0
    sumG = 0
    sumB = 0
    for i in range(0, 7):
        for j in range(0, 7):
            sumR += src[tidx-3+i, tidy-3+j, 0]*filter[i, j]
            sumG += src[tidx-3+i, tidy-3+j, 1]*filter[i, j]
            sumB += src[tidx-3+i, tidy-3+j, 2]*filter[i, j]
    dst[tidx-3, tidy-3, 0] = np.float32(sumR/1003)
    dst[tidx-3, tidy-3, 1] = np.float32(sumG/1003)
    dst[tidx-3, tidy-3, 2] = np.float32(sumB/1003)


@cuda.jit
def gaussian_gpu_share(src, dst, filter):
    tile = cuda.shared.array((7, 7), np.uint8)
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if cuda.threadIdx.x < 7 and cuda.threadIdx.y < 7:
        tile[cuda.threadIdx.x,
             cuda.threadIdx.y] = filter[cuda.threadIdx.x, cuda.threadIdx.y]

    cuda.syncthreads()

    sumR = 0
    sumG = 0
    sumB = 0
    for i in range(0, 7):
        for j in range(0, 7):
            sumR += src[tidx-3+i, tidy-3+j, 0]*filter[i, j]
            sumG += src[tidx-3+i, tidy-3+j, 1]*filter[i, j]
            sumB += src[tidx-3+i, tidy-3+j, 2]*filter[i, j]
    dst[tidx-3, tidy-3, 0] = np.float32(sumR/1003)
    dst[tidx-3, tidy-3, 1] = np.float32(sumG/1003)
    dst[tidx-3, tidy-3, 2] = np.float32(sumB/1003)




blockSize_range = [(2,2),(4,4),(8,8),(16,16),(32,32)]
for blockSize in blockSize_range:
  gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))
  devOutput = cuda.device_array((numpy_img.shape), np.float64)
  devInput = cuda.to_device(numpy_img)
  time1 = time.time()
  gaussian_gpu[gridSize, blockSize](devInput, devOutput, filter)
  hostOutput = devOutput.copy_to_host()
  print('GPU time in seconds of blocksize ',blockSize,' : ', time.time()-time1)


for blockSize in blockSize_range:
  gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))
  devOutput = cuda.device_array((numpy_img.shape), np.float64)
  devInput = cuda.to_device(numpy_img)
  time1 = time.time()
  gaussian_gpu_share[gridSize, blockSize](devInput, devOutput, filter)
  hostOutput = devOutput.copy_to_host()
  print('GPU time in seconds of blocksize ',blockSize,' : ', time.time()-time1)




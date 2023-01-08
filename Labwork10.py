from numba import cuda, jit
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import math

img = Image.open('000054.JPG')
#img = img.resize((786,528))
#img = img.resize((393,264))
img.save('sample.png')




numpy_img = np.asarray(img)
numpy_img = numpy_img.astype('float64')
numpy_img = numpy_img/255


window_size = 4
numpy_img = np.pad(numpy_img, [(window_size,window_size),(window_size,window_size),(0,0)])
print(numpy_img.shape)


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

def KuwaharaCPU(img_RGB, img_HSV, w):
    finalrgb = np.zeros((img_RGB.shape))

    for x in range(w, img_RGB.shape[0]-w):
        for y in range(w,img_RGB.shape[1]-w):
            range_List = (((x-w, x+1), (y-w, y+1)),
                        ((x, x+w+1), (y-w, y+1)),
                        ((x-w, x+1), (y, y+w+1)),
                        ((x, x+w+1), (y, y+w+1)))
            minid = 0
            minstd = 256

            for window in range(4):
                win_sum = 0

                for i in range(*range_List[window][0]):
                    for j in range(*range_List[window][1]):
                        win_sum += img_HSV[2, i, j]

                n = (w+1)**2
                winMean = win_sum/n
                win_sum = 0

                for i in range(*range_List[window][0]):
                    for j in range(*range_List[window][1]):
                        win_sum += (img_HSV[2, i, j] - winMean)**2 
                win_std = math.sqrt(win_sum/n)
                if win_std < minstd:
                    minstd = win_std
                    minid = window

            range_min = range_List[minid]
            avg = 0
            for i in range(*range_min[0]):
                for j in range(*range_min[1]):
                    avg += img_RGB[i, j]
            finalrgb[x, y] = avg / (w+1)**2
    return finalrgb


@ cuda.jit
def KuwaharaGPU(srcRGB, srcHSV, dst, w):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if tidx < w or tidx >= srcRGB.shape[0] - w or tidy < w or tidy >= srcRGB.shape[1] - w:
        dst[tidx, tidy, 0] = dst[tidx, tidy, 1] = dst[tidx, tidy, 2] = 0
        return


    minid = 0
    minstd = 256
    range_List = (((tidx-w, tidx+1), (tidy-w, tidy+1)),
                ((tidx, tidx+w+1), (tidy-w, tidy+1)),
                ((tidx-w, tidx+1), (tidy, tidy+w+1)),
                ((tidx, tidx+w+1), (tidy, tidy+w+1)))

    for window in range(4):
        win_sum = 0

        for i in range(*range_List[window][0]):
            for j in range(*range_List[window][1]):
                win_sum += srcHSV[2, i, j]

        n = (w+1)**2
        winMean = win_sum/n
        win_sum = 0

        for i in range(*range_List[window][0]):
            for j in range(*range_List[window][1]):
                win_sum += (srcHSV[2, i, j] - winMean)**2 
        win_std = math.sqrt(win_sum/n)
        if win_std < minstd:
            minstd = win_std
            minid = window

    range_min = range_List[minid]

    avg_r = 0
    avg_g = 0
    avg_b = 0
    for i in range(*range_min[0]):
        for j in range(*range_min[1]):
            avg_r += srcRGB[i, j, 0]
            avg_g += srcRGB[i, j, 1]
            avg_b += srcRGB[i, j, 2]

    dst[tidx, tidy, 0] = avg_r / (w+1)**2
    dst[tidx, tidy, 1] = avg_g / (w+1)**2
    dst[tidx, tidy, 2] = avg_b / (w+1)**2



blocksize = (2,2)
gridSize = (math.ceil(width/blocksize[0]), math.ceil(height/blocksize[1]))
HSVOutput = cuda.device_array((3, width, height), np.float64)
devInput = cuda.to_device(numpy_img)
RGB2HSV[gridSize, blocksize](devInput, HSVOutput)
HSV_out = HSVOutput.copy_to_host()


time1 = time.time()
imgcpu = KuwaharaCPU(numpy_img, HSV_out, window_size)
print('CPU time in seconds: ', time.time()-time1)



blockSize_range = [(2,2),(4,4),(8,8),(16,16),(32,32)] #
for blockSize in blockSize_range:
  gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))
  KuraOutput = cuda.device_array((numpy_img.shape), np.float64)
  devInput = cuda.to_device(numpy_img)
  time1 = time.time()

  KuwaharaGPU[gridSize, blockSize](devInput,HSVOutput, KuraOutput,window_size)
  imggpu = KuraOutput.copy_to_host()
  print('GPU time in seconds of blocksize ',blockSize,' : ', time.time()-time1)


Ori_img = numpy_img
plt.imshow(Ori_img)
plt.show()

plt.imshow(imgcpu)
plt.show()

plt.imshow(imggpu)
plt.show()

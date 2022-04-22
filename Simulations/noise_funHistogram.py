# Here I am going to copy the noise_fun code from Primary_Simulation_1.py and make a histogram out of noise_fun to 
# see if it looks correct.

zhiyuanRange = False

import numpy as np
from numpy.random import standard_normal
from PIL import Image
import matplotlib.pyplot as plt

imdim = 1024

nnoise = 1
noisefact = 16

assert imdim % noisefact == 0
noise_kernel_size = int(imdim/noisefact)    # 256 ->32, 512 ~ 32, 1024 -> 128   # Comment meaning

assert imdim % noise_kernel_size == 0
resize_factor = int(imdim/noise_kernel_size)

noise_fn = np.zeros((noise_kernel_size, noise_kernel_size))

if zhiyuanRange:

    for i in range(nnoise):
        noise_fn += np.random.uniform(-1, 0.6, size=(noise_kernel_size, noise_kernel_size))

else:

    for i in range(nnoise):
        noise_fn += standard_normal(size=(noise_kernel_size, noise_kernel_size))

noise_fn /= nnoise
noise_fn += 1
noise_fn /= 2

new_shape = tuple([resize_factor * i for i in noise_fn.shape])
noise_fun = np.array(Image.fromarray(noise_fn).resize(size=new_shape))

# noise_funHistogram = np.histogram(noise_fun)

# See documentation of matplotlib.pyplot.hist for what the below (especially n, bins, patches) means
n, bins, patches = plt.hist(noise_fun)

plt.xlabel('Value of element of noise_fun')
plt.ylabel('Number of elements with this value')
plt.title('Histogram of noise_fun')
plt.show()
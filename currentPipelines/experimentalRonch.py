# Made this pipeline to process experimental Ronchigrams found as single images in .dm3 files in the directory 
# /media/rob/hdd1/james-gj/forReport/2022-04-29/2022-04-29
# 
# The parent directory OF THE ABOVE (/media/rob/hdd1/james-gj/forReport/2022-04-29) contains:
# 
# -> /media/rob/hdd1/james-gj/forReport/2022-04-29/20220429_Ronchigram.xlsx, which has information about the dosage for 
#    acquiring each Ronchigram etc.
# 
# -> /media/rob/hdd1/james-gj/forReport/2022-04-29/cosmo.txt, which has aberration values etc.
# 
# See https://docs.google.com/document/d/1_X-KUGV4fcsEkN-lzOrHu8TAVfcSy-TemvELxcnsV5Q/edit for some crucial information 
# pertaining to the acquisition of these Ronchigrams.

import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL
from ncempy.io import dm
import imageio

with h5py.File(f'/media/rob/hdd1/james-gj/forReport/2022-04-29/experimentalRonchs.h5', 'w') as f:

    # 0: HDF5 datasets

    try:
        random_mags_dset = f.create_dataset("random_mags dataset", (number_processes, simulations_per_process, 14), dtype="float32")
        random_angs_dset = f.create_dataset("random_angs dataset", (number_processes, simulations_per_process, 14), dtype="float32")
        random_I_dset = f.create_dataset("random_I dataset", (number_processes, simulations_per_process, 1), dtype="float32")
        random_t_dset = f.create_dataset("random_t dataset", (number_processes, simulations_per_process, 1), dtype="float32")
        random_seed_dset = f.create_dataset("random_seed dataset", (number_processes, simulations_per_process, 1), dtype="int")
        ronch_dset = f.create_dataset("ronch dataset", (number_processes, simulations_per_process, 1024, 1024), dtype="float32")


    except:
        random_mags_dset = f["random_mags dataset"]
        random_angs_dset = f["random_angs dataset"]
        random_I_dset = f["random_I dataset"]
        random_t_dset = f["random_t dataset"]
        random_seed_dset = f["random_seed dataset"]
        ronch_dset = f["ronch dataset"]


    # 1: Read from /media/rob/hdd1/james-gj/forReport/2022-04-29/2022-04-29 the data, including (most crucially) the image 
    # arrays themselves, converting them into numpy form.

    # This was me attempting to read the data using dm3, however, I don't really know what to do with the array returned, 
    # specifically its negative values, especially since it is already float32.
    # img = dm.dmReader('/media/rob/hdd1/james-gj/forReport/2022-04-29/2022-04-29/Orius SC600A 2_20kX_0001.dm3')
    # imgData = img['data']
    # imgData = np.asarray(imgData)
    # print(np.amin(imgData))
    # imgData -= np.amin(imgData)
    # imgData /= np.amax(imgData)
    # plt.imshow(imgData)
    # plt.show()

    # Went with the below because the below is already in uint8, to which it is typically converted in RonchigramDataset 
    # class definition.
    # imgData2 = imageio.imread('/media/rob/hdd1/james-gj/forReport/2022-04-29/PNG Files/Orius SC600A 2_20kX_0001.png')

    # uint8 dtype before and after making into numpy array; the object type beforehand was image.core.util.array, a subclass 
    # of numpy.ndarray
    # imgData2 = np.asarray(imgData2)


    # 3: Normalise the numpy arrays as is done in Primary_Simulation_1.py
    # I probably need to discuss this process (and the nature of the dm3 files) more with Chen or another member of the 
    # research group. However, for now, I am not doing any further normalisation. Not sure if I should do subtraction like 
    # in simulations because, in simulations, subtraction was not actually done because there are elements in the array that 
    # are zero (and that is minimum) due to mask. Moreover, here, some values are negative.


    # 4: Read the aberration constants etc. from /media/rob/hdd1/james-gj/forReport/2022-04-29/cosmo.txt, which has 
    # aberration values etc.

    # Going to set up HDF5 file





    # 5: Read from /media/rob/hdd1/james-gj/forReport/2022-04-29 other data that is typically saved with simulations made in 
    # Parallel_HDF5_2.py, like noise values etc.

    # 6: Save the loaded data to HDF5 the same way that this is done in Parallel_HDF5_2.py, so that RonchigramDataset can 
    # access it the same way without you having to modify the definition of this class; however, if not possible, modification
    # or the creation of a new class is indeed allowed.
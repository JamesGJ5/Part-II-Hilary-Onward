import h5py
from random import randint
import numpy as np
import mpi4py

with h5py.File("/media/rob/hdd2/james/simulations/20_01_22/Single_Aberrations.h5", "r") as f:
    random_mags_dset = f["random_mags dataset"]
    random_angs_dset = f["random_angs dataset"]
    random_I_dset = f["random_I dataset"]
    random_t_dset = f["random_t dataset"]
    ronch_dset = f["ronch dataset"]

    for dset in [random_mags_dset, random_angs_dset, random_I_dset, random_t_dset, ronch_dset]:
        print("\n", dset)

    for rank in range(ronch_dset.shape[0]):
        # Remember, rank is basically the process ID in Parallel_HDF5_2.py
        print(ronch_dset[rank, 0])
        print(ronch_dset[rank, -1])
    
    # print(dset[0, :])

    # print(ronch_dset.shape)
    # print(ronch_dset.shape[0])

    # print(random_mags_dset[5545:5555, :])

    # random_ronchigram = ronch_dset[78321, :]
    # It seems that, for some reason, after the Ronchigram at the 5555 index, the Ronchigram elements are all 0 for some reason
    # The Log says 5556 simulations were done per process so maybe it is something to do with that - perhaps things were 
    # only working for one process. Maybe, however, if things are stored in HDF5, they are still stored on the CPU so I have 
    # to access the core in which they are stored somehow. That could be the issue I am having.
    # It seems that the webpage dosc.h5py.org/en/stable/mpi.html may be of service
    # However, perhaps it is also possible that the same 5555 Ronhigrams continue to get overwritten. To check, I will print 
    # a bunch of sets of aberration magnitudes, and see which magnitudes are being saved. Actually, this wouldn't yield 
    # an answer, since for each process, Ronchigrams of different types of aberration will be simulated. Instead, I will 
    # continue with the Parallel HDF5 attempt. Having issues installing mpi4py in the conda environment "pytorch", going 
    # to create a new conda environment and attempt to use that. Now using pytorch3
    # print(random_ronchigram)
    # print(np.amax(random_ronchigram))
    # print(np.amax(random_ronchigram) == 0)
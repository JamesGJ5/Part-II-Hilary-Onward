import h5py
from random import randint
import numpy as np
import mpi4py

for file_index in range(18):
    with h5py.File(f"/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/Single_Aberrations_{file_index}.h5") as f:
        random_mags_dset = f["random_mags dataset"]
        random_angs_dset = f["random_angs dataset"]
        random_I_dset = f["random_I dataset"]
        random_t_dset = f["random_t dataset"]
        ronch_dset = f["ronch dataset"]

        for dset in [random_mags_dset, random_angs_dset, random_I_dset, random_t_dset, ronch_dset]:
            print("\n", dset)
            print(dset[49, :])

        # The result of this is not just filled with zero, suggesting all saved well. Now just have to merge the files together.
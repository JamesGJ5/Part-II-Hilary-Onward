import h5py
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle

# with h5py.File(f'/media/rob/hdd1/james-gj/forReport/2022-04-29/experimentalRonchigrams.h5', 'r', driver='mpio', comm=MPI.COMM_WORLD) as f:

    # dose_pct_dset = f['dose_pct dataset']

    # for dose_pct in dose_pct_dset:

    #     print(dose_pct)


    # comso_t_s_dset = f['cosmo_t_s dataset']

    # for cosmo_t_s in comso_t_s_dset:

    #     print(cosmo_t_s)


    # orius_t_s_dset = f['random_t dataset']

    # for orius_t_s in orius_t_s_dset:

    #     print(orius_t_s)


    # mags_dset = f['random_mags dataset']

    # for mag in mags_dset:

    #     print(mag)


    # angs_dset = f['random_angs dataset']

    # for ang in angs_dset:

    #     print(ang)

    # magErrors_dset = f['magnitude_error_unknownUnit dataset']

    # for magError in magErrors_dset:

    #     print(magError)


    # angErrors_dset = f['angle_error_unknownUnit dataset']

    # for angError in angErrors_dset:

    #     print(angError)


    # pi_over_4_limit_dset = f['pi_over_4_limit_in_m dataset']

    # for pi_over_4_limit_in_m in pi_over_4_limit_dset:

    #     print(pi_over_4_limit_in_m)


    # ronch_dset = f['ronch dataset']

    # print(ronch_dset)

    # for ronch in ronch_dset[0, :]:

    #     print(ronch.shape)

    #     plt.imshow(ronch, cmap='gray')
    #     plt.show()

    # pixel_size_dset = f['pixel_size dataset']

    # print(pixel_size_dset)

    # for pixelSize in pixel_size_dset[0, :]:

    #     print(pixelSize)

f = open('/home/james/VSCode/currentPipelines/max_magList.pkl', 'rb')
max_magList = pickle.load(f)
f.close()

print(max_magList)
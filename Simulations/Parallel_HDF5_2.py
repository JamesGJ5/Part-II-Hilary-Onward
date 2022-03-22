import random
import Primary_Simulation_1
import numpy as np
import h5py
import math
import time
import os
from mpi4py import MPI
import cmath
import sys

from datetime import datetime

if __name__ == "__main__":

    # Must check whether this is really necessary by testing simulating without it
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # Output Ronchigram will be an array of size imdim x imdim in pixels
    imdim = 1024

    # simdim is essentially the convergence semi-angle (or maximum tilt angle) in rad. It was called simdim in code by Hovden Labs so I 
    # do the same because the below is for simulations of Ronchigrams done on the basis of code adapted from them.
    simdim = 50 * 10**-3

    # The maxima below apply when making Ronchigrams in which the aberration in question is to be significant
    max_C10 = 10 * 10**-9  # Maximum C10 (defocus) magnitude/m
    max_C12 = 10 * 10**-9  # Maximum C12 (2-fold astigmatism) magnitude/m
    max_C21 = 1000 * 10**-9  # Maximum C21 (axial coma) magnitude/m
    max_C23 = 1000 * 10**-9  # Maximum C23 (3-fold astigmatism) magnitude/m

    # min_Cnm will be 0 since negative values are redundant, I THINK (see lab book's 29/11/2021 entry)
    # phi_n,m will be between 0 and pi/m radians since, I believe, other angles are redundant (see lab book's 29/11/2021 entry)

    # NOISE PARAMETER VALUES
    # See lab book entry 02/12/2021

    min_I = 100 * 10**-12   # Minimum quoted current/A
    max_I = 1 * 10**-9      # Maximum quoted current/A
    b = 1   # i.e. assuming that all of quoted current in Ronchigram acquisition reaches the detector
    min_t = 0.1 # Minimum Ronchigram acquisition time/s
    max_t = 1   # Maximum Ronchigram acquisition time/s

    def simulate_single_aberrations(number_simulations: int, imdim: int, simdim: float, max_C10: float, 
        max_C12: float, max_C21: float, max_C23: float, min_I: float, max_I: float, 
        min_t: float, max_t: float) -> None:
        # TODO: take this outside of function because, unlike the multiprocessing method I found in a video, MPI 
        # doesn't need the script being parallelised to be put in a function definition.
        """Simulates single-aberration Ronchigrams and saves them along with the magnitudes and angles individually. 
        min_Cnm = 0 (for dominant aberration), b = 1, and phi_n,m are random between 0 and pi radians (except for phi10).

        :param number_simulations: number of Ronchigrams simulated per process
        :param imdim: Ronchigram array size in pixels (imdim x imdim)
        :param simdim: effectively Ronchigram convergence semi-angle/rad
        :param max_C10: max. defocus magnitude/m
        :param max_C12: max. 2-fold astigmatism/m
        :param max_C21: max. axial coma/m
        :param max_C23: max. 3-fold astigmatism/m
        """

        with h5py.File(f"/media/rob/hdd1/james-gj/Simulations/forTraining/22_03_22/mixedAbers.h5", "w", driver="mpio", comm=MPI.COMM_WORLD) as f:
            # Be wary that you are in write mode

            # TODO: code in a way to add the value(s) of b to the HDF5 file if you choose to
            try:
                # dtype is float64 rather than float32 to reduce the memory taken up in storage.
                random_mags_dset = f.create_dataset("random_mags dataset", (number_processes, number_simulations, 5), dtype="float32")
                random_angs_dset = f.create_dataset("random_angs dataset", (number_processes, number_simulations, 5), dtype="float32")
                random_I_dset = f.create_dataset("random_I dataset", (number_processes, number_simulations, 1), dtype="float32")
                random_t_dset = f.create_dataset("random_t dataset", (number_processes, number_simulations, 1), dtype="float32")
                ronch_dset = f.create_dataset("ronch dataset", (number_processes, number_simulations, 1024, 1024), dtype="float32")
            
            except:
                random_mags_dset = f["random_mags dataset"]
                random_angs_dset = f["random_angs dataset"]
                random_I_dset = f["random_I dataset"]
                random_t_dset = f["random_t dataset"]
                ronch_dset = f["ronch dataset"]

            randu = random.uniform

            # Initialising simulation_number variable that will be incremented below
            # NOTE: The below variable is only useful for certain statements below
            # simulation_number = 0

            linearC10 = np.linspace(max_C10 + rank / number_processes * max_C10, max_C10 + (rank + 1) / number_processes * max_C10, number_simulations, endpoint=False)
            linearC12 = np.linspace(max_C12 + rank / number_processes * max_C12, max_C12 + (rank + 1) / number_processes * max_C12, number_simulations, endpoint=False)
            linearC21 = np.linspace(max_C21 + rank / number_processes * max_C21, max_C21 + (rank + 1) / number_processes * max_C21, number_simulations, endpoint=False)
            linearC23 = np.linspace(max_C23 + rank / number_processes * max_C23, max_C23 + (rank + 1) / number_processes * max_C23, number_simulations, endpoint=False)

            linearPhi12 = np.linspace(rank / number_processes * np.pi/2, (rank + 1) / number_processes * np.pi/2, number_simulations, endpoint=False)
            linearPhi21 = np.linspace(rank / number_processes * np.pi, (rank + 1) / number_processes * np.pi, number_simulations, endpoint=False)
            linearPhi23 = np.linspace(rank / number_processes * np.pi/3, (rank + 1) / number_processes * np.pi/3, number_simulations, endpoint=False)

            # See Google doc 4th Year > 16/02/22 for how the below ranges were chosen
            for simulation in range(number_simulations):
                # NOTE: The below variable is only useful for certain statements below
                # simulation_number += 1

                # C10 = randu(max_C10, 2 * max_C10)
                # C12 = randu(max_C12, 2 * max_C12)
                # C21 = randu(max_C21, 2 * max_C21)
                # C23 = randu(max_C23, 2 * max_C23)

                C10 = randu(0, max_C10)
                C12 = randu(0, max_C12)
                C21 = randu(0, max_C21)
                C23 = randu(0, max_C23)
                C30 = 0

                # C10 = linearC10[simulation]
                # C12 = 50 * 10**-9
                # C21 = 5000 * 10**-9
                # C23 = 5000 * 10**-9

                # phi10 = 0
                # phi12 = np.pi/4
                # phi21 = np.pi/2
                # phi23 = np.pi/6

                # if simulation_number <= math.ceil(number_simulations / 4):
                #     C10 = randu(0, max_C10)

                #     C12 = randu(0, C10/100)
                #     C21 = randu(0, C10/10)
                #     C23 = randu(0, C10/10)

                # elif math.ceil(number_simulations / 4) < simulation_number <= math.ceil(number_simulations / 2):
                #     C12 = randu(0, max_C12)

                #     C10 = randu(0, C12/100)
                #     C21 = randu(0, C12/10)
                #     C23 = randu(0, C12/10)

                # elif math.ceil(number_simulations / 2) < simulation_number <= math.ceil(3 * number_simulations / 4):
                #     C21 = randu(0, max_C21)
                    
                #     C10 = randu(0, C21/1000)
                #     C12 = randu(0, C21/1000)
                #     C23 = randu(0, C21/100)

                # elif math.ceil(3 * number_simulations / 4) < simulation_number:
                #     C23 = randu(0, max_C23)
                    
                #     C10 = randu(0, C23/1000)
                #     C12 = randu(0, C23/1000)
                #     C21 = randu(0, C23/100)

                # Below, the ranges for 
                phi10 = 0   # Defocus has an m-value of 0
                phi12 = randu(0, np.pi / 2)
                phi21 = randu(0, np.pi / 1)
                phi23 = randu(0, np.pi / 3)
                phi30 = 0

                I = randu(min_I, max_I)
                t = randu(min_t, max_t)

                # Note: simulations have toe be saved as below for the dataloader in DataLoader2.py to work, so be 
                # careful when it comes to making changes to the below. Also, make sure that the spaces created in the 
                # HDF5 file are filled completely, otherwise the length method of the dataset won't work properly, and 
                # nor will the getitem method.
                random_mags = np.array([C10, C12, C21, C23, C30])
                random_mags_dset[rank, simulation] = random_mags[:]

                random_angs = np.array([phi10, phi12, phi21, phi23, phi30])
                random_angs_dset[rank, simulation] = random_angs[:]

                random_I = np.array([I])
                random_I_dset[rank, simulation] = random_I[:]

                random_t = np.array([t])
                random_t_dset[rank, simulation] = random_t[:]

                ronch = Primary_Simulation_1.calc_Ronchigram(imdim, simdim, C10, C12, C21, C23, C30, phi10, phi12, phi21, phi23, phi30, I, b, t)
                ronch_dset[rank, simulation] = ronch[:]

                # FIXME: the below is a good way of keeping track of what is happening but it was applicable to the 
                # multiprocessing things, not the MPI stuff. May recreate it for MPI.

                # To make sure things are running properly (only want to do for one process lest we get an overflowing terminal)
                # if processnum == 0:
                #     if simulation_number == 1:
                #         print("\n")
                #         print(random_mags_dset[simulation])
                #         print(random_angs_dset[simulation])
                #         print(random_I_dset[simulation])
                #         print(ronch_dset[simulation])

                #     if simulation_number % 100 == 0:
                #         tested_at = time.perf_counter()
                #         time_to_test = round(tested_at - start, 2)
                #         print(f"\n{simulation_number} simulations complete for process number {processnum}")
                #         print(f"{time_to_test} seconds elapsed since script began running")

    # This code was here to make a small practice file (/media/rob/hdd2/james/simulations/17_01_22/Single_Aberrations_Lite.h5)
    # number_processes = 1
    # rank = 0
    # simulate_single_aberrations(10, imdim, simdim, max_C10, max_C12, max_C21, max_C23, min_I, max_I, min_t, max_t)

    # sys.exit()

    # CPUs AND PROCESSES
    total_simulations = 100000

    number_processes = MPI.COMM_WORLD.size
    simulations_per_process = int(math.ceil(total_simulations / number_processes))

    rank = MPI.COMM_WORLD.rank
    # The below was in use because previously I was writing HDF5 files containg space for simulations from all 32 channels 
    # while not using all 32 of them - this led to unnecessary space being consumed, so I just used number_processes in 
    # creating my datasets instead.
    # channels = 32


    # START TIME METRICS
    start = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print("\nStarted at:", start)
    start = time.perf_counter() # Another start variable, computed to later facilitate printing how long the processes took


    # CALLING THE SIMULATION FUNCTION ABOVE
    simulate_single_aberrations(simulations_per_process, imdim, simdim, max_C10, max_C12, max_C21, max_C23, min_I, max_I, min_t, max_t)

    # Finish time metrics
    finish_date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print("\nFinished at:", finish_date_time)

    finish = time.perf_counter()
    time_taken_overall = f"Finished in {round(finish - start, 2)} seconds"

    fout = open("Log", "a")
    fout.write(f"\n\nFinished at: {finish_date_time}")
    fout.write(f"\n{time_taken_overall}")
    fout.write(f"\n{total_simulations} simulations done, {number_processes} processes used, {simulations_per_process} simulations per process")
    fout.close()
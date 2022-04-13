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
    simdim = 70 * 10**-3

    # Essentially the convergence semi-angle/mrad; only called aperture_size because objective aperture size controls this 
    # quantity and wanted to be consistent with (Schnitzer, 2020c) in Primary_Simulation_1.py
    aperture_size = simdim

    # The maxima below apply when making Ronchigrams in which the aberration in question is to be significant
    max_C10 = 100 * 10**-9  # Maximum C10 (defocus) magnitude/m
    max_C12 = 100 * 10**-9  # Maximum C12 (2-fold astigmatism) magnitude/m

    max_C21 = 300 * 10**-9  # Maximum C21 (axial coma) magnitude/m
    max_C23 = 100 * 10**-9  # Maximum C23 (3-fold astigmatism) magnitude/m

    max_C30 = 10.4 * 10**-6
    max_C32 = 10.4 * 10**-6
    max_C34 = 5.22 * 10**-6

    max_C41 = 0.1 * 10**-3
    max_C43 = 0.1 * 10**-3
    max_C45 = 0.1 * 10**-3

    max_C50 = 10 * 10**-3
    max_C52 = 10 * 10**-3
    max_C54 = 10 * 10**-3
    max_C56 = 10 * 10**-3

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
        
        with h5py.File(f"/media/rob/hdd1/james-gj/Simulations/forInference/_/simdim70mrad/randC12linPhi12_randOthers.h5", "w", driver="mpio", comm=MPI.COMM_WORLD) as f:
            # Be wary that you are in write mode

            # TODO: code in a way to add the value(s) of b to the HDF5 file if you choose to
            try:
                # dtype is float64 rather than float32 to reduce the memory taken up in storage.
                random_mags_dset = f.create_dataset("random_mags dataset", (number_processes, number_simulations, 14), dtype="float32")
                random_angs_dset = f.create_dataset("random_angs dataset", (number_processes, number_simulations, 14), dtype="float32")
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
            simulation_number = 0

            # linearC10 = np.linspace(rank / number_processes * max_C10, (rank + 1) / number_processes * max_C10, number_simulations, endpoint=False)
            linearC12 = np.linspace(rank / number_processes * max_C12, (rank + 1) / number_processes * max_C12, number_simulations, endpoint=False)
            # linearC21 = np.linspace(rank / number_processes * max_C21, (rank + 1) / number_processes * max_C21, number_simulations, endpoint=False)
            # linearC23 = np.linspace(rank / number_processes * max_C23, (rank + 1) / number_processes * max_C23, number_simulations, endpoint=False)

            linearPhi12 = np.linspace(rank / number_processes * 2*np.pi/2, (rank + 1) / number_processes * 2*np.pi/2, number_simulations, endpoint=False)
            # linearPhi21 = np.linspace(rank / number_processes * 2*np.pi, (rank + 1) / number_processes * 2*np.pi, number_simulations, endpoint=False)
            # linearPhi23 = np.linspace(rank / number_processes * 2*np.pi/3, (rank + 1) / number_processes * 2*np.pi/3, number_simulations, endpoint=False)

            # See Google doc 4th Year > 16/02/22 for how the below ranges were chosen
            for simulation in range(number_simulations):
                # Just for line 187 (i.e. status updates)
                simulation_number += 1

                C10 = randu(0, max_C10)
                # C10 = max_C10 / 2
                C12 = randu(0, max_C12)
                # C12 = linearC12[simulation]

                C21 = randu(0, max_C21)
                # C21 = max_C21 / 2
                C23 = randu(0, max_C23)
                # C23 = max_C23 / 2

                C30 = randu(0, max_C30)
                # C30 = max_C30 / 2
                C32 = randu(0, max_C32)
                # C32 = max_C32 / 2
                C34 = randu(0, max_C34)
                # C34 = max_C34 / 2

                C41 = randu(0, max_C41)
                # C41 = max_C41 / 2
                C43 = randu(0, max_C43)
                # C43 = max_C43 / 2
                C45 = randu(0, max_C45)
                # C45 = max_C45 / 2

                C50 = randu(0, max_C50)
                # C50 = max_C50 / 2
                C52 = randu(0, max_C52)
                # C52 = max_C52 / 2
                C54 = randu(0, max_C54)
                # C54 = max_C54 / 2
                C56 = randu(0, max_C56)
                # C56 = max_C56 / 2


                phi10 = 0
                # phi12 = randu(0, 2 * np.pi / 2)
                phi12 = linearPhi12[simulation]

                phi21 = randu(0, 2 * np.pi / 1)
                # phi21 = 2 * np.pi / 2
                phi23 = randu(0, 2 * np.pi / 3)
                # phi23 = 2 * np.pi / 6

                phi30 = 0
                phi32 = randu(0, 2 * np.pi / 2)
                # phi32 = 2 * np.pi / 4
                phi34 = randu(0, 2 * np.pi / 4)
                # phi34 = 2 * np.pi / 8

                phi41 = randu(0, 2 * np.pi / 1)
                # phi41 = 2 * np.pi / 2
                phi43 = randu(0, 2 * np.pi / 3)
                # phi43 = 2 * np.pi / 6
                phi45 = randu(0, 2 * np.pi / 5)
                # phi45 = 2 * np.pi / 10

                phi50 = 0
                phi52 = randu(0, 2 * np.pi / 2)
                # phi52 = 2 * np.pi / 4
                phi54 = randu(0, 2 * np.pi / 4)
                # phi54 = 2 * np.pi / 8
                phi56 = randu(0, 2 * np.pi / 6)
                # phi56 = 2 * np.pi / 12

                I = randu(min_I, max_I)
                t = randu(min_t, max_t)

                # Note: simulations have toe be saved as below for the dataloader in DataLoader2.py to work, so be 
                # careful when it comes to making changes to the below. Also, make sure that the spaces created in the 
                # HDF5 file are filled completely, otherwise the length method of the dataset won't work properly, and 
                # nor will the getitem method.
                random_mags = np.array([C10, C12, C21, C23, C30, C32, C34, C41, C43, C45, C50, C52, C54, C56])
                random_mags_dset[rank, simulation] = random_mags[:]

                random_angs = np.array([phi10, phi12, phi21, phi23, phi30, phi32, phi34, phi41, phi43, phi45, phi50, phi52, phi54, phi56])
                random_angs_dset[rank, simulation] = random_angs[:]

                random_I = np.array([I])
                random_I_dset[rank, simulation] = random_I[:]

                random_t = np.array([t])
                random_t_dset[rank, simulation] = random_t[:]

                ronch = Primary_Simulation_1.calc_Ronchigram(imdim, simdim,
                                                            C10, C12, C21, C23, C30, C32, C34, C41, C43, C45, C50, C52, C54, C56,
                                                            phi10, phi12, phi21, phi23, phi30, phi32, phi34, phi41, phi43, phi45, 
                                                            phi50, phi52, phi54, phi56,
                                                            I, b, t,
                                                            aperture_size=aperture_size)
                ronch_dset[rank, simulation] = ronch[:]


                if simulation_number % math.ceil(number_simulations / 40) == 0:

                    print(f"\n{simulation_number} simulations complete for rank at index {rank} at " + \
                            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")


    # CPUs AND PROCESSES
    total_simulations = 1000

    number_processes = MPI.COMM_WORLD.size
    simulations_per_process = int(math.ceil(total_simulations / number_processes))

    rank = MPI.COMM_WORLD.rank


    # START TIME METRICS
    start = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print("\nStarted at:", start)
    start = time.perf_counter() # Another start variable, computed to later facilitate printing how long the processes took


    # CALLING THE SIMULATION FUNCTION ABOVE
    simulate_single_aberrations(simulations_per_process, imdim, simdim, max_C10, max_C12, max_C21, max_C23, min_I, max_I, min_t, max_t)


    # FINISH TIME METRICS
    finish_date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print("\nFinished at:", finish_date_time)

    finish = time.perf_counter()
    time_taken_overall = f"Finished in {round(finish - start, 2)} seconds"

    fout = open("Log", "a")
    fout.write(f"\n\nFinished at: {finish_date_time}")
    fout.write(f"\n{time_taken_overall}")
    fout.write(f"\n{total_simulations} simulations done, {number_processes} processes used, {simulations_per_process} simulations per process")
    fout.close()
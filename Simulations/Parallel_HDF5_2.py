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
from numpy.random import default_rng
import pickle

from datetime import datetime

if __name__ == "__main__":

    # Must check whether this is really necessary by testing simulating without it
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # This is the file from which you may be intending to copy certain parameters. For example, might be wanting to 
    # copy just magnitudes and angles (where applicable) for C10-C23 in the function below, but change simdim, for 
    # example
    mimicFile = False

    if mimicFile:
        
        mimickedFile = input("Input the path of the file to be mimicked: ")

    # Output Ronchigram will be an array of size imdim x imdim in pixels
    imdim = 1024

    # simdim is essentially the convergence semi-angle (or maximum tilt angle) in rad. It was called simdim in code by Hovden Labs so I 
    # do the same because the below is for simulations of Ronchigrams done on the basis of code adapted from them.
    simdim = 80 * 10**-3

    # Essentially the convergence semi-angle/mrad; only called aperture_size because objective aperture size controls this 
    # quantity and wanted to be consistent with (Schnitzer, 2020c) in Primary_Simulation_1.py
    aperture_size = simdim

    f = open('/home/james/VSCode/currentPipelines/max_magList.pkl', 'rb')
    max_magList = pickle.load(f)
    f.close()

    max_C10, max_C12, max_C21, max_C23, max_C30, max_C32, max_C34, max_C41, max_C43, max_C45, max_C50, max_C52, max_C54, max_C56 = max_magList

    print(max_C10 / 10**-9, max_C12 / 10**-9, max_C21 / 10**-9, max_C23 / 10**-9, \
    max_C30 / 10**-6, max_C32 / 10**-6, max_C34 / 10**-6, \
    max_C41 / 10**-6, max_C43 / 10**-6, max_C45 / 10**-6, \
    max_C50 / 10**-3, max_C52 / 10**-3, max_C54 / 10**-3, max_C56 / 10**-3)

    # The maxima below for C10-C23 apply when making Ronchigrams in which the aberration in question is to be significant
    # max_C10 = 100 * 10**-9  # Maximum C10 (defocus) magnitude/m
    # max_C12 = 100 * 10**-9  # Maximum C12 (2-fold astigmatism) magnitude/m

    # max_C21 = 300 * 10**-9  # Maximum C21 (axial coma) magnitude/m
    # max_C23 = 100 * 10**-9  # Maximum C23 (3-fold astigmatism) magnitude/m

    # max_C30 = 10.4 * 10**-6
    # max_C32 = 10.4 * 10**-6
    # max_C34 = 5.22 * 10**-6

    # max_C41 = 0.1 * 10**-3
    # max_C43 = 0.1 * 10**-3
    # max_C45 = 0.1 * 10**-3

    # max_C50 = 10 * 10**-3
    # max_C52 = 10 * 10**-3
    # max_C54 = 10 * 10**-3
    # max_C56 = 10 * 10**-3

    # min_Cnm will be 0 since negative values are redundant, I THINK (see lab book's 29/11/2021 entry)
    # phi_n,m will be between 0 and pi/m radians since, I believe, other angles are redundant (see lab book's 29/11/2021 entry)

    # NOISE PARAMETER VALUES
    # See lab book entry 02/12/2021

    min_I = 100 * 10**-12   # Minimum quoted current/A
    max_I = 1 * 10**-9      # Maximum quoted current/A
    b = 1   # i.e. assuming that all of quoted current in Ronchigram acquisition reaches the detector
    min_t = 0.1 # Minimum Ronchigram acquisition time/s
    max_t = 1   # Maximum Ronchigram acquisition time/s


    def simulate_single_aberrations(simulations_per_process: int, imdim: int, simdim: float, min_I: float, 
        max_I: float, min_t: float, max_t: float, saveFile: str, mimickedFile = None) -> None:
        # TODO: take this outside of function because, unlike the multiprocessing method I found in a video, MPI 
        # doesn't need the script being parallelised to be put in a function definition.
        """Simulates single-aberration Ronchigrams and saves them along with the magnitudes and angles individually. 
        min_Cnm = 0 (for dominant aberration), b = 1, and phi_n,m are random between 0 and pi radians (except for phi10).

        :param simulations_per_process: number of simulations to be made in each process
        :param imdim: Ronchigram array size in pixels (imdim x imdim)
        :param simdim: effectively Ronchigram convergence semi-angle/rad
        :param max_C10: max. defocus magnitude/m
        :param max_C12: max. 2-fold astigmatism/m
        :param max_C21: max. axial coma/m
        :param max_C23: max. 3-fold astigmatism/m
        """
        
        if mimicFile:

            fMimic = h5py.File(mimickedFile, "r")

            randMags = fMimic["random_mags dataset"]
            randAngs = fMimic["random_angs dataset"]
            randI = fMimic["random_I dataset"]
            randt = fMimic["random_t dataset"]
            randSeed = fMimic["random_seed dataset"]

            simulations_per_process = randMags.shape[1]

        with h5py.File(saveFile, "x", driver="mpio", comm=MPI.COMM_WORLD) as f:

            # print(f"Simulations per process is {simulations_per_process}")

            # TODO: code in a way to add the value(s) of b to the HDF5 file if you choose to
            try:
                # dtype is float64 rather than float32 to reduce the memory taken up in storage.
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

            randu = random.uniform

            # Initialising simulation_number variable that will be incremented below
            # NOTE: The below variable is only useful for certain statements below
            simulation_number = 0

            # linearC10 = np.linspace(rank / number_processes * max_C10, (rank + 1) / number_processes * max_C10, simulations_per_process, endpoint=False)
            linearC12 = np.linspace(rank / number_processes * max_C12, (rank + 1) / number_processes * max_C12, simulations_per_process, endpoint=False)
            # linearC21 = np.linspace(rank / number_processes * max_C21, (rank + 1) / number_processes * max_C21, simulations_per_process, endpoint=False)
            # linearC23 = np.linspace(rank / number_processes * max_C23, (rank + 1) / number_processes * max_C23, simulations_per_process, endpoint=False)

            linearPhi12 = np.linspace(rank / number_processes * 2*np.pi/2, (rank + 1) / number_processes * 2*np.pi/2, simulations_per_process, endpoint=False)
            # linearPhi21 = np.linspace(rank / number_processes * 2*np.pi, (rank + 1) / number_processes * 2*np.pi, simulations_per_process, endpoint=False)
            # linearPhi23 = np.linspace(rank / number_processes * 2*np.pi/3, (rank + 1) / number_processes * 2*np.pi/3, simulations_per_process, endpoint=False)

            # See Google doc 4th Year > 16/02/22 for how the below ranges were chosen
            for simulation in range(simulations_per_process):
                # Just for line 187 (i.e. status updates)
                simulation_number += 1

                # C10 = 0
                C10 = randu(0, max_C10)
                # C10 = max_C10 / 2
                # C10 = randMags[rank, simulation, 0]

                # C12 = 0
                C12 = randu(0, max_C12)
                # C12 = linearC12[simulation]
                # C12 = randMags[rank, simulation, 1]

                # C21 = 0
                C21 = randu(0, max_C21)
                # C21 = max_C21 / 2
                # C21 = randMags[rank, simulation, 2]

                # C23 = 0
                C23 = randu(0, max_C23)
                # C23 = max_C23 / 2
                # C23 = randMags[rank, simulation, 3]

                # C30 = 0
                C30 = randu(0, max_C30)
                # C30 = max_C30 / 2

                # C32 = 0
                C32 = randu(0, max_C32)
                # C32 = max_C32 / 2

                # C34 = 0
                C34 = randu(0, max_C34)
                # C34 = max_C34 / 2

                # C41 = 0
                C41 = randu(0, max_C41)
                # C41 = max_C41 / 2

                # C43 = 0
                C43 = randu(0, max_C43)
                # C43 = max_C43 / 2

                # C45 = 0
                C45 = randu(0, max_C45)
                # C45 = max_C45 / 2

                # C50 = 0
                C50 = randu(0, max_C50)
                # C50 = max_C50 / 2

                # C52 = 0
                C52 = randu(0, max_C52)
                # C52 = max_C52 / 2

                # C54 = 0
                C54 = randu(0, max_C54)
                # C54 = max_C54 / 2

                # C56 = 0
                C56 = randu(0, max_C56)
                # C56 = max_C56 / 2


                phi10 = 0

                # phi12 = 0
                # phi12 = randu(0, 2 * np.pi / 2)
                phi12 = linearPhi12[simulation]
                # phi12 = randAngs[rank, simulation, 1]

                # phi21 = 0
                phi21 = randu(0, 2 * np.pi / 1)
                # phi21 = 2 * np.pi / 2
                # phi21 = randAngs[rank, simulation, 2]

                # phi23 = 0
                phi23 = randu(0, 2 * np.pi / 3)
                # phi23 = 2 * np.pi / 6
                # phi23 = randAngs[rank, simulation, 3]

                phi30 = 0

                # phi32 = 0
                phi32 = randu(0, 2 * np.pi / 2)
                # phi32 = 2 * np.pi / 4

                # phi34 = 0
                phi34 = randu(0, 2 * np.pi / 4)
                # phi34 = 2 * np.pi / 8

                # phi41 = 0
                phi41 = randu(0, 2 * np.pi / 1)
                # phi41 = 2 * np.pi / 2

                # phi43 = 0
                phi43 = randu(0, 2 * np.pi / 3)
                # phi43 = 2 * np.pi / 6

                # phi45 = 0
                phi45 = randu(0, 2 * np.pi / 5)
                # phi45 = 2 * np.pi / 10

                phi50 = 0

                # phi52 = 0
                phi52 = randu(0, 2 * np.pi / 2)
                # phi52 = 2 * np.pi / 4

                # phi54 = 0
                phi54 = randu(0, 2 * np.pi / 4)
                # phi54 = 2 * np.pi / 8

                # phi56 = 0
                phi56 = randu(0, 2 * np.pi / 6)
                # phi56 = 2 * np.pi / 12

                if not mimicFile:

                    # random_seed = 17
                    random_seed = None

                    I = default_rng(random_seed).uniform(min_I, max_I)
                    t = default_rng(random_seed).uniform(min_t, max_t)

                else:

                    random_seed = int(randSeed[rank, simulation])

                    I = randI[rank, simulation, 0]
                    t = randt[rank, simulation, 0]

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

                # Want to limit the chance of multiple simulations having the same seed.
                # TODO: make disimilar random seeds

                random_seed = chosenSeeds[simulation]
                random_seed_dset[rank, simulation] = random_seed

                ronch = Primary_Simulation_1.calc_Ronchigram(imdim, simdim,
                                                            C10, C12, C21, C23, C30, C32, C34, C41, C43, C45, C50, C52, C54, C56,
                                                            phi10, phi12, phi21, phi23, phi30, phi32, phi34, phi41, phi43, phi45, 
                                                            phi50, phi52, phi54, phi56,
                                                            I, b, t,
                                                            aperture_size=aperture_size, seed=random_seed)
                ronch_dset[rank, simulation] = ronch[:]


                if simulation_number % math.ceil(simulations_per_process / 60) == 0:

                    print(f"\n{simulation_number} simulations complete for rank at index {rank} at " + \
                            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")

        if mimicFile:

            fMimic.close()


    # CPUs AND PROCESSES

    number_processes = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    
    if not mimicFile:
    
        total_simulations = 1000
        simulations_per_process = int(math.ceil(total_simulations / number_processes))

        # print(f"Simulations per process is {simulations_per_process}")

        # Factor of 10 is included below for good measure; really, for replace=False in default_rng().choice, possibleSeeds 
        # must only contain as many elements as there are Ronchigrams in the process, but there shouldn't be an issue with 
        # having more.
        possibleSeeds = np.arange(rank * simulations_per_process * 10, (rank + 1) * simulations_per_process * 10)

        # Don't want to repeat a seed for different Ronchigrams, hence replace=False.
        chosenSeeds = default_rng().choice(possibleSeeds, len(possibleSeeds), replace=False)

    else:

        simulations_per_process = 0
        print("A file is being mimicked")


    # START TIME METRICS
    start = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print("\nStarted at:", start)
    start = time.perf_counter() # Another start variable, computed to later facilitate printing how long the processes took


    # CALLING THE SIMULATION FUNCTION ABOVE
    # Here, X are just the maximum percentages of c1,2 in the range being sampled from; the corresponding minimum % is 
    # (X - 10)
    # XList = [10 * (i + 1) for i in range(10)]

    # for X in XList:

    #     saveFileX = f'/media/rob/hdd1/james-gj/Simulations/forInference/30_05_22/c12_{X-10}to{X}pct.h5'

    #     currentMin_C12 = max_C12 * (X - 10) / 100
    #     currentMax_C12 = max_C12 * X / 100

    #     print(currentMin_C12, currentMax_C12)

    #     simulate_single_aberrations(simulations_per_process=simulations_per_process, imdim=imdim, simdim=simdim, 
    #     max_C10=max_C10, max_C12=currentMax_C12, max_C21=max_C21, max_C23=max_C23, min_I=min_I, max_I=max_I, min_t=min_t, 
    #     max_t=max_t, saveFile=saveFileX, min_C12=currentMin_C12)

    saveFile = '/media/rob/hdd1/james-gj/Simulations/forInference/17_06_22/randC12linPhi12_randOthers.h5'

    simulate_single_aberrations(simulations_per_process=simulations_per_process, imdim=imdim, simdim=simdim, 
        min_I=min_I, max_I=max_I, min_t=min_t, max_t=max_t, saveFile=saveFile)


    # FINISH TIME METRICS
    finish_date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print("\nFinished at:", finish_date_time)

    # finish = time.perf_counter()
    # time_taken_overall = f"Finished in {round(finish - start, 2)} seconds"

    # fout = open("Log", "a")
    # fout.write(f"\n\nFinished at: {finish_date_time}")
    # fout.write(f"\n{time_taken_overall}")
    # fout.write(f"\n{number_processes * simulations_per_process} simulations done, {number_processes} processes used")
    # fout.write(f"\n{simulations_per_process} simulations per process")

    # fout.close()
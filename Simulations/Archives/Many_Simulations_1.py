import random

# FIXME: now that Many_Simulations_1.py is in a sub-directory in the directory in which Primary_Simulation_1.py is found 
# the below import statement probably won't work.
import Primary_Simulation_1
import numpy as np
import h5py
import math
import time
import multiprocessing

from datetime import datetime

if __name__ == "__main__":

    # Output Ronchigram will be an array of size imdim x imdim in pixels
    imdim = 1024
    # simdim is essentially the convergence semi-angle (or maximum tilt angle) in rad. It was called simdim in code by Hovden Labs so I 
    # do the same because the below is for simulations of Ronchigrams done on the basis of code adapted from them.
    simdim = 100 * 10**-3
    max_C10 = 100 * 10**-9  # Maximum C10 (defocus) magnitude/m
    max_C12 = 100 * 10**-9  # Maximum C12 (2-fold astigmatism) magnitude/m
    max_C21 = 600 * 10**-9  # Maximum C21 (axial coma) magnitude/m
    max_C23 = 760 * 10**-9  # Maximum C23 (3-fold astigmatism) magnitude/m

    # min_Cnm will be 0 since negative values are redundant, I THINK (see lab book's 29/11/2021 entry)

    # NOTE: THE BELOW LINE IS INCORRECT, IT IS ACTUALLY BETWEEN 0 AND PI/m RADIANS, AS THE LAB BOOK ENTRY MENTIONED BELOW NOW SAYS. I WILL 
    # NOT CHANGE THE CODE SINCE IT IS ARCHIVED CODE BUT BE CAREFUL.
    # phi_n,m will be between 0 and pi radians since other angles are redundant, I THINK (see lab book's 29/11/2021 entry)

    # NOISE PARAMETER VALUES
    # See lab book entry 02/12/2021

    min_I = 100 * 10**-12   # Minimum quoted current/A
    max_I = 1 * 10**-9      # Maximum quoted current/A
    b = 1   # i.e. assuming that all of quoted current in Ronchigram acquisition reaches the detector
    min_t = 0.1 # Minimum Ronchigram acquisition time/s
    max_t = 1   # Maximum Ronchigram acquisition time/s

    def simulate_single_aberrations(number_simulations: int, imdim: int, simdim: int or float, max_C10: int or float, 
        max_C12: int or float, max_C21: int or float, max_C23: int or float, min_I: int or float, max_I: int or float, 
        min_t: int or float, max_t: int or float) -> None:
        """A function defined to run simulations for single-aberration Ronchigrams until I can find out how to 
        use the multiprocessing library (or another) for concurrent CPU processes without needing to use a 
        function. This is done using min_Cnm = 0, b = 0 and with random phi_n,m between 0 and pi radians.

        :param number_simulations: total number of Ronchigrams being simulated
        :param imdim: output Ronchigram array size in pixels (imdim x imdim)
        :param simdim: effectively the Ronchigram's convergence semi-angle/rad
        :param max_C10: maximum defocus magnitude/m
        :param max_C12: maximum 2-fold astigmatism/m
        :param max_C21: maximum axial coma/m
        :param max_C23: maximum 3-fold astigmatism/m
        """

        global process
        process += 1
        print(f"\nProcess number {process} is running")

        randu = random.uniform

        simulation_number = 0
        for simulation in range(number_simulations):
            simulation_number += 1

            if simulation_number <= math.ceil(number_simulations / 4):
                C10 = randu(0, max_C10)
                C12 = 0
                C21 = 0
                C23 = 0

            elif math.ceil(number_simulations / 4) < simulation_number <= math.ceil(number_simulations / 2):
                C10 = 0
                C12 = randu(0, max_C12)
                C21 = 0
                C23 = 0

            elif math.ceil(number_simulations / 2) < simulation_number <= math.ceil(3 * number_simulations / 4):
                C10 = 0
                C12 = 0
                C21 = randu(0, max_C21)
                C23 = 0

            elif math.ceil(3 * number_simulations / 4) < simulation_number:
                C10 = 0
                C12 = 0
                C21 = 0
                C23 = randu(0, max_C23)


            # NOTE: THE BELOW LINE IS INCORRECT, IT IS ACTUALLY BETWEEN 0 AND PI/m RADIANS, AS THE 29/11/21 LAB BOOK NOW SAYS. I WILL 
            # NOT CHANGE THE CODE SINCE IT IS ARCHIVED CODE BUT BE CAREFUL.
            phi10 = 0   # Defocus has an m-value of 0
            phi12 = randu(0, np.pi)
            phi21 = randu(0, np.pi)
            phi23 = randu(0, np.pi)

            I = randu(min_I, max_I)
            t = randu(min_t, max_t)

            random_mags = np.array([C10, C12, C21, C23])
            random_mags_dset[simulation] = random_mags[:]

            random_angs = np.array([phi10, phi12, phi21, phi23])
            random_angs_dset[simulation] = random_angs[:]

            random_I = np.array([I])
            random_I_dset[simulation] = random_I[:]

            random_t = np.array([t])
            random_t_dset[simulation] = random_t[:]

            ronch = Primary_Simulation_1.calc_Ronchigram(imdim, simdim, C10, C12, C21, C23, phi10, phi12, phi21, phi23, I, b, t)
            ronch_dset[simulation] = ronch[:]

            # To make sure things are running properly (only want to do for one process lest we get an overflowing terminal)
            if process == 1:
                if simulation_number == 1:
                    print("\n")
                    print(random_mags_dset[simulation])
                    print(random_angs_dset[simulation])
                    print(random_I_dset[simulation])
                    print(ronch_dset[simulation])

                if simulation_number % 100 == 0:
                    tested_at = time.perf_counter()
                    time_to_test = round(tested_at - start, 2)
                    print(f"\n{simulation_number} simulations complete for process number {process}")
                    print(f"{time_to_test} seconds elapsed since script began running")

    with h5py.File("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/Single_Aberrations.h5", "a") as f:
    # Be wary that you are in append mode

        # INITIALISING HDF5 DATASE
        number_simulations = 100000

        random_mags_dset = f.create_dataset("random_mags dataset", (number_simulations, 4), dtype="float64")
        random_angs_dset = f.create_dataset("random_angs dataset", (number_simulations, 4), dtype="float64")
        random_I_dset = f.create_dataset("random_I dataset", (number_simulations, 1), dtype="float64")
        random_t_dset = f.create_dataset("random_t dataset", (number_simulations, 1), dtype="float64")
        # TODO: make sure you have used the correct shape for the random_I_dset and random_t_dset dataset
        ronch_dset = f.create_dataset("ronch dataset", (number_simulations, 1024, 1024), dtype="float64")

        # CPUs AND PROCESSES
        num_of_cpu = multiprocessing.cpu_count()
        print(f"\nNumber of CPUs available: {num_of_cpu}")
        number_processes = 18
        print(f"Number of processes to be used: {number_processes}")
        simulations_per_process = math.ceil(number_simulations / number_processes)
        print(f"{simulations_per_process} simulations will be done per process")

        # START TIME METRICS
        start = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        print("\nStarted at:", start)
        start = time.perf_counter() # Another start variable, computed to later facilitate printing how long the processes took

        # MULTIPROCESSING
        processes = []

        # I am putting this here so that inside the function definition I can increment process by 1 and print some 
        # statements only for when process == 1
        process = 0

        for process in range(number_processes):
            p = multiprocessing.Process(target=simulate_single_aberrations(simulations_per_process, imdim, simdim, 
                max_C10, max_C12, max_C21, max_C23, min_I, max_I, min_t, max_t))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        # Finish time metrics
        finish_date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        print("\nFinished at:", finish_date_time)

        finish = time.perf_counter()
        time_taken_overall = f"Finished in {round(finish - start, 2)} seconds"

        fout = open("Log", "a")
        fout.write(f"\n\nFinished at: {finish_date_time}")
        fout.write(f"\n{time_taken_overall}")
        fout.write(f"\n{number_simulations} simulations done, {number_processes} processes used, {simulations_per_process} simulations per process")
        fout.close()

        # Check how much time is taken for 1000 Ronchigrams (took about 6 minutes, so 100000 Ronchigrams will probably 
        # take about 10 hours)
        # Check how much space gets taken up by 1000 Ronchigrams (at most 8.9GB, so 100000 Ronchigrams will probably take 
        # up at most 890GB - I say at most because after 1000 Ronchigrams were simulated the file 
        # /media/rob/hdd1/james-gj/Ronchigrams/Simulations/06_12_2021/Single_Aberrations.h5 had a nominal size of 8.9GB, 
        # but in the current simulation it seems )
        # This (i.e. checking memory) will also check if your code is running multi-core
        # Sometimes going to fewer cores helps with speed, sometimes it is the 
        #   RAM usage, sometimes it's the cache
        # Try to find a good way to document what simulations you have done (i.e. documenting what the scripts do for 
        #   example)

        # N.B.: Ronchigrams stored will not have been normalised

        # Must find a good way to provide metrics to keep track of where things are in each process and check that each 
        # process really is running concurrently.
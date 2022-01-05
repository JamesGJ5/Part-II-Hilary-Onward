Directory made for generating simulations to train the networks with.

Notable Contents:

    Primary_Simulation_1.py is for Schnitzer's Ronchigram simulations.

    Simulation_Loading_1.py was made to check if the first set of simulations generated were working correctly, i.e. 
    the simulations where I assumed if I ran multiple processes they would all save to the same HDF5 file without me 
    enabling parallel HDF5. I then altered it to test if parallel simulations using MPI with parallel HDF5 enabled (i.e. 
    into a single HDF5 file) was working.

    Parallel_HDF5_2.py was made to generate simulations in parallel (using MPI) and store them in the same HDF5 file, 
    after having enabled parallel HDF5 via a process mentioned in my Google doc, My Drive > 4th Year > Resources 
    > HDF5 > Parallel HDF5

Archived Contents:

    Secondary_Simulation_1.py is for attempts at making Ronchigram simulations from (Sawada et al., 2008)

    Simulation_Loading_2.py was made to check if the simulations made not using parallel HDF5, but saving different 
    processes to different files, in hopes of stitching these files together later.

    Merging.py was made to merge together the HDF5 files. This turned into putting links for each of the 18 HDF5 files 
    into a new HDF5 file, using advice from https://stackoverflow.com/questions/58187004/how-can-i-combine-multiple-h5-file/58293398#58293398 and 
    https://stackoverflow.com/questions/60782932/is-there-a-way-to-quickly-extract-specified-tables-into-a-different-hdf5-file/60792094#60792094, 
    beginning with the former since it seems more applicable.

    Merging_Tester.py was made to test if the 18 files had been merged successfully. It did this by loading the simulations 
    from the merged HDF5 file and seeing if they looked OK.

    Multiprocessing_Practice.py was a file I made to practise using Python's multiprocessing libraries to allow me to 
    carry out simulations in multiple processes. Of course, I am now using MPI and Parallel HDF5 instead.

    Many_Simulations_1.py was the first file I created to generate simulations on mass, although not in parallel and not 
    being saved to HDF5. Later, I attempted to save in parallel to HDF5, but this didn't work because HDF5 was not enabled, 
    so I made Parallel_HDF5.py

    As said, Parallel_HDF5.py was made for multiprocessing, this time to load simulations into 18 different folders for later 
    stitching.

...

Extra information:

    11:01am on 01/01/2022: have just created Parallel_HDF5_2.py for what I think should be working simulations for saving in 
    parallel to a single HDF5 file. I am going to copy over the code from Parallel_HDF5.py rather than Many_Simulations_1.py, 
    since I think the former is what I used to make the last set of simulations.
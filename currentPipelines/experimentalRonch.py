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




# 1: Read from /media/rob/hdd1/james-gj/forReport/2022-04-29/2022-04-29 the data, including (most crucially) the image 
# arrays themselves, converting them into numpy form.

# 2: Normalise the numpy arrays as is done in Primary_Simulation_1.py

# 3: Read the aberration constants etc. from /media/rob/hdd1/james-gj/forReport/2022-04-29/cosmo.txt, which has 
# aberration values etc.

# 4: Read from /media/rob/hdd1/james-gj/forReport/2022-04-29 other data that is typically saved with simulations made in 
# Parallel_HDF5_2.py, like noise values etc.

# 5: Save the loaded data to HDF5 the same way that this is done in Parallel_HDF5_2.py, so that RonchigramDataset can 
# access it the same way without you having to modify the definition of this class; however, if not possible, modification
# or the creation of a new class is indeed allowed.
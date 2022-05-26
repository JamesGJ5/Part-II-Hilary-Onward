import h5py
from mpi4py import MPI
import pandas as pd
import numpy as np
import pickle
import sys
from math import radians
from ncempy.io import dm
import matplotlib.pyplot as plt
import math
import cv2

# Creating this file to conveniently group together each experimental Ronchigram with properties like aberration 
# constants, acquisition times, noise levels etc, so that inference (particularly predicting c1,2 and phi1,2) can be 
# done on these easily and predicted Ronchigrams can be plotted alongside test Ronchigrams.

# The experimental Ronchigrams I have so far are those that Chen acquired on 29/04/2022, which can be found as: 
# 
# -> .dm3 files in /media/rob/hdd1/james-gj/forReport/2022-04-29/2022-04-29
# 
# -> Since conversion using ImageJ, .png files in /media/rob/hdd1/james-gj/forReport/2022-04-29/PNG Files

# The parent directory of both of the above (/media/rob/hdd1/james-gj/forReport/2022-04-29) contains:
# 
# -> /media/rob/hdd1/james-gj/forReport/2022-04-29/20220429_Ronchigram.xlsx, which has information about the dosage for 
#    acquiring each Ronchigram etc.
# 
# -> /media/rob/hdd1/james-gj/forReport/2022-04-29/cosmo.txt, which has aberration values etc.

# Since processing in txtRead.py, aberCalcResults_2022_04_29_OPQ.pkl currently have all of the aberration information from comso.txt in 
# the following format:
# 
# -> d1 = {aberrationCalculationTime: aberrationCalculationResult}
# 
# -> aberrationCalculationResult = {aberrationType: aberrationParameters}, were aberrationType is either O2 (C1,0), A2 (C1,2), P3 (C2,1) etc.
# 
# -> aberrationParameters = {
#                           'mag': aberration magnitude in magUnit,
#                           'magUnit': unit in which 'mag' is found,
#                           'magError': value that is, ACCORDING TO ZIYI, error associated with mag BUT I MUST DISCOVER ITS UNIT,
#                           'angle': aberration angle (presumably in OPQ form) in angleUnit,
#                           'angleUnit': unit in which 'angle' is found,
#                           'angleError': value that is, ACCORDING TO ZIYI, error associated with angle BUT I MUST DISCOVER ITS UNIT,
#                           'pi/4Limit': I NEED TO ASK CHEN ABOUT THIS but it is a value in pi/4LimitUnit,
#                           'pi/4LimitUnit: unit in which 'pi/4Limit' limit is found
# }

# I want to save the experimental Ronchigrams, along with their aberration constants, in HDF5 format as I have done with  
# the simulations I have used so far--this will make it easy to integrate these with the inference pipelines I already 
# have, particularly comparisonInferencer.py but I could probably also make trend graphs out of these, just the 
# c1,2 and phi1,2 labels wouldn't vary between Ronchigrams in an exactly linear fashion.

# Since I am not exactly sure yet (I am waiting for Chen to tell me) which aberration constants are for which image 
# besides the aberration constants whose recording times are listed in 
# /media/rob/hdd1/james-gj/forReport/2022-04-29/20220429_Ronchigram.xlsx, for now I will just save to HDF5, along with 
# their parameters like aberration constants, noise, capture time etc., the experimental Ronchigrams that are explicitly 
# mentioned in the aforementioned file


# 0: Read the .xlsx file containing the image acquisition parameters

acquisitionParams = pd.read_excel(
    io='/media/rob/hdd1/james-gj/forReport/2022-04-29/20220429_Ronchigram.xlsx', 
    header=0,
    usecols="C:E, G, H",
    skiprows=[4, 6, 9, 12],
    dtype=str
    )
# print(acquisitionParams)

acquisitionParamsDict = acquisitionParams.to_dict()
# print(acquisitionParamsDict)

# Just a dictionary of idx: Image (number), which will be used later when choosing the image to save to a certain idx of the HDF5 file
idxImageNumberDict = acquisitionParamsDict['Image']


# 1. Create the HDF5 datasets as in Parallel_HDF5_2.py; keep the number of processes sort of thing, although this will 
# probably just be given a value of 1
# -> Implement a way to save dose/% (if you haven't yet found out how to convert to current); Cosmo t (s); each aberration 
#    magnitude' error (identified by Ziyi), although Chen hasn't yet verified this or told me its unit; each aberration angle's 
#    error (identified by Ziyi), although Chen hasn't yet verified this or told me its unit; pi/4 limit in METRES for 
#    each aberration

with h5py.File(f'/media/rob/hdd1/james-gj/forReport/2022-04-29/experimentalRonchigrams.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:

    number_processes = 1
    simulations_per_process = len(acquisitionParamsDict['Image'])

    try:
        # dtype is float64 rather than float32 to reduce the memory taken up in storage.
        random_mags_dset = f.create_dataset("random_mags dataset", (number_processes, simulations_per_process, 14), dtype="float32")
        random_angs_dset = f.create_dataset("random_angs dataset", (number_processes, simulations_per_process, 14), dtype="float32")
        random_I_dset = f.create_dataset("random_I dataset", (number_processes, simulations_per_process, 1), dtype="float32")
        random_t_dset = f.create_dataset("random_t dataset", (number_processes, simulations_per_process, 1), dtype="float32")
        random_seed_dset = f.create_dataset("random_seed dataset", (number_processes, simulations_per_process, 1), dtype="int")
        ronch_dset = f.create_dataset("ronch dataset", (number_processes, simulations_per_process, 1024, 1024), dtype="float32")

        dose_pct_dset = f.create_dataset("dose_pct dataset", (number_processes, simulations_per_process, 1), dtype="float32")
        comso_t_s_dset = f.create_dataset("cosmo_t_s dataset", (number_processes, simulations_per_process, 1), dtype="float32")
        magnitude_error_unknownUnit_dset = f.create_dataset("magnitude_error_unknownUnit dataset", (number_processes, simulations_per_process, 14), dtype="float32")
        angle_error_unknownUnit_dset = f.create_dataset("angle_error_unknownUnit dataset", (number_processes, simulations_per_process, 14), dtype="float32")
        pi_over_4_limit_in_m_dset = f.create_dataset("pi_over_4_limit_in_m dataset", (number_processes, simulations_per_process, 14), dtype="float32")
        pixel_size_dset = f.create_dataset('pixel_size dataset', (number_processes, simulations_per_process, 2), dtype='float32')

    except:
        random_mags_dset = f["random_mags dataset"]
        random_angs_dset = f["random_angs dataset"]
        random_I_dset = f["random_I dataset"]
        random_t_dset = f["random_t dataset"]
        random_seed_dset = f["random_seed dataset"]
        ronch_dset = f["ronch dataset"]

        dose_pct_dset = f['dose_pct dataset']
        comso_t_s_dset = f['comso_t_s dataset']
        magnitude_error_unknownUnit_dset = f['magnitude_error_unknownUnit dataset']
        angle_error_unknownUnit_dset = f['angle_error_unknownUnit dataset']
        pi_over_4_limit_in_m_dset = f['pi_over_4_limit_in_m dataset']
        pixel_size_dset = f['pixel_size dataset']


    # Probably going to have a for loop for each time a file is being read--don't really want to open a file over and over 
    # again


    # 2. Reading the images and saving them to ronch_dset
    # -> Use the values of idxImageNumberDict above to find the image numbers (present in .dm3 file names) of the Ronchigrams to be saved; 
    #    extract the Ronchigrams from the .dm3 files whose names feature said numbers and save them to the [0, idx] (idx being the corresponding 
    #    key of idxImageNumberDict) position of ronch_dset

    # The images are numpy arrays of type float32. Orius SC600A 2_20kX_0001.dm3 has size 2688x2672. The images are 2D (there is no colour 
    # channel).

    # im0 = dm.dmReader('/media/rob/hdd1/james-gj/forReport/2022-04-29/2022-04-29/Orius SC600A 2_20kX_0001.dm3')
    # print("Data has been read, I think")
    # imgArray = im0['data']
    # pxSize = im0['pixelSize']
    # print(pxSize)
    # plt.imshow(imgArray, cmap='gray', vmin=0, vmax=np.amax(imgArray))
    # plt.show()

    for idx, imageNumber in idxImageNumberDict.items():

        # 1. Loading the image array itself using dm.dmReader then accessing the 'data' key from the resulting dictionary

        imgContents = dm.dmReader(f"/media/rob/hdd1/james-gj/forReport/2022-04-29/2022-04-29/Orius SC600A 2_20kX_00{imageNumber.rjust(2, '0')}.dm3")
        
        pixelSizeData = imgContents['pixelSize']
        pixel_size_dset[0, idx] = np.array(pixelSizeData)

        imgArray = imgContents['data']


        # 2. Clip the array such that negative elements are converted to zero

        imgArray = imgArray.clip(min=0)


        # 3. Crop the array such that the borders of the array are tangential to the circle in which the Ronchigrammy bit exists.

        # Got these indices from looking at Orius SC600A 2_20kX_0001.dm3 in ImageJ
        imgArray = imgArray[116:2544, 132:2536]


        # 4. If the image is not a square at this point, but a x b pixels where a is smaller than b, centercrop the image to a square of 
        # length a

        shape = imgArray.shape

        cropSquareLength = min(shape)

        assert imgArray.ndim == 2
        imgArray = imgArray[
            math.ceil(shape[0] / 2) - math.ceil(cropSquareLength / 2): math.ceil(shape[0] / 2) + math.floor(cropSquareLength / 2),
            math.ceil(shape[0] / 2) - math.floor(cropSquareLength / 2): math.ceil(shape[0] / 2) + math.floor(cropSquareLength / 2)
            ]

        shape = imgArray.shape

        # N.B: since clipping is done beforehand, vmin=0 isn't strictly needed to display the Ronchigram realistically, 
        # but I keep it here anyway
        # plt.imshow(imgArray, cmap='gray', vmin=0, vmax=np.amax(imgArray))
        # plt.show()


        # 5. Resize image to 1024 x 1024 but fail if initial image is smaller than this (want to be wary if I am upsizing since I have not 
        # yet done this)

        assert shape[0] >= 1024
        
        # Here, I chose inter_cubic because it is a bicubic interpolation, and I use a bicubic interpolation in my 
        # Resize() transform in training.py
        imgArray = cv2.resize(imgArray, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)

        # N.B: since clipping is done beforehand, vmin=0 isn't strictly needed to display the Ronchigram realistically, 
        # but I keep it here anyway
        # plt.imshow(imgArray, cmap='gray', vmin=0, vmax=np.amax(imgArray))
        # plt.show()


        # 6. The part where the image is actually saved to HDF5

        ronch_dset[0, idx] = imgArray[:]

        print(f"Image number {imageNumber} saved...")


    # 3. Make sure the normalization of the above is adequate
    # TODO: use DataLoader2.py to figure this out


    # 4. Save dose/% alongside the above Ronchigrams

    idxDosePctDict = acquisitionParamsDict['Dose(%)']

    for idx, dosePct in idxDosePctDict.items():
        
        dosePct = np.array([eval(dosePct)])

        dose_pct_dset[0, idx] = dosePct


    # 5. Converting dose/% to current/A and save alongside the above Ronchigrams
    # NOTE: will have to wait for Chen to give me the current corresponding to 100% dose, which he will probably be able 
    # to do on Monday 30/05/22


    # 6. Saving cosmo t (s), which I still don't really know how to descirbe yet, alongside each Ronchigram

    idxCosmo_t_sDict = acquisitionParamsDict['Cosmo t (s)']

    for idx, cosmo_t_s in idxCosmo_t_sDict.items():

        cosmo_t_s = np.array([eval(cosmo_t_s)])

        comso_t_s_dset[0, idx] = cosmo_t_s


    # 7. Saving Orius t (s), which I believe is Ronchigram capture time/s needed for Poisson noise recreation, alongside 
    # each Ronchigram

    idxOrius_t_sDict = acquisitionParamsDict['Orius t (s)']

    for idx, orius_t_s in idxOrius_t_sDict.items():

        orius_t_s = np.array([eval(orius_t_s)])

        random_t_dset[0, idx] = orius_t_s


    # 8. For each time in the list from step 1, remembering to save these constants to position imageIdx in the HDF5 dataset, 
    # get the aberration parameters recorded at this time:

    idxTimeDict = acquisitionParamsDict['Time']

    with open('/home/james/VSCode/currentPipelines/aberCalcResults_2022_04_29_OPQ.pkl', 'rb') as f:

        allResultSets = pickle.load(f)

        # These are the coefficient of the theta angles in "Measurement method of aberration from Ronchigra by 
        # autocorrelation function" by Sawada et al.. N.B., there are no unique angles for the O aberrations (they are 
        # rotationally symmetric) so the 1 is arbitrary in those cases but it is unused; for the rest, the coefficient 
        # will be later multiplied by respective aberration angles to get these in Krivanek notation; I assume the 
        # errors are have the same units as what they are evaluating error for so this is indeed done.
        # TODO: if the errors are instead percentage errors, but in degrees or rad, then get rid of the multiplication
        aberAngleCoefficientDict = {'O2': 1, 'A2': 2, 'P3': 1, 'A3': 3, 'O4': 1, 'Q4': 2, 'A4': 4, 'P5': 1, 'R5': 3, 
                                    'A5': 5, 'O6': 1, 'Q6': 2, 'S6': 4, 'A6': 6}

        unitsToBaseUnitsIn_m = {'nm': 10**-9, 'um': 10**-6, 'mm': 10**-3}

        for idx, timeOfAcquisition in idxTimeDict.items():

            mags = np.array([])
            angs = np.array([])
            
            magErrors = np.array([])
            angErrors = np.array([])
            pi_over_4_limits_in_m = np.array([])

            # The 1: because in the .xlsx file there's a 0 at the beginning of each time whereas there's not in the 
            # file cosmo.txt
            aberCalcSet = allResultSets[timeOfAcquisition[1:]]

            for aber in aberAngleCoefficientDict:

                aberParams = aberCalcSet[aber]


                # -> Aberration magnitude/m

                magUnit = aberParams['magUnit']

                if magUnit in unitsToBaseUnitsIn_m:

                    magIn_m = eval(aberParams['mag']) * unitsToBaseUnitsIn_m[magUnit]
                    mags = np.append(mags, magIn_m)

                else:

                    sys.exit('Magnitude was not initially saved in nm, um, or mm; must add another multiplier to the above to' + \
                    'account for the unit it has in this case.')


                # -> Aberration angle/degree (will have to convert to radians then multiply by whatever needs multiplying by to get the 
                #    Krivanek notation aberration angle value)

                if aberParams['angleUnit'] == 'degree':

                    angInRad = radians(eval(aberParams['angle']))
                    krivanekAngInRad = angInRad * aberAngleCoefficientDict[aber]

                    angs = np.append(angs, angInRad)

                else:

                    sys.exit('Angle was not initially saved in degrees; must add another multiplier to the above to' + \
                    'account for the unit it has in this case.')


                # -> Aberration magnitude error/unknown unit

                # TODO: I believe the aberration magnitude errors in cosmo.txt are in the same units as the magnitudes 
                # because I think the aberration angle errors are in degrees for reasons given below. But, if not, I must
                # disable the following two lines and just append eval(aberParams['magError']) directly to angErrors 
                # instead
                magErrorInMagUnit = eval(aberParams['magError'])
                magErrorIn_m = magErrorInMagUnit * unitsToBaseUnitsIn_m[magUnit]

                magErrors = np.append(magErrors, magErrorIn_m)


                # -> Aberration angle error/unknown unit

                # TODO: I believe the aberration angle errors in cosmo.txt are in degrees because of how large the values are 
                # but, if not, I must disable the following two lines and just append eval(aberParams['angleError']) 
                # directly to angErrors instead
                angErrorDeg = eval(aberParams['angleError'])
                krivanekAngErrorRad = radians(angErrorDeg) * aberAngleCoefficientDict[aber]

                angErrors = np.append(angErrors, krivanekAngErrorRad)


                # -> pi/4 limit in metres

                # TODO: if it turns out pi/4 limit must be converted from OPQ to Krivanek notation or something, amend the 
                # below
                piOver4LimitInUnit = eval(aberParams['pi/4Limit'])

                piOver4LimitUnit = aberParams['pi/4LimitUnit']
                piOver4LimitIn_m = piOver4LimitInUnit * unitsToBaseUnitsIn_m[piOver4LimitUnit]

                pi_over_4_limits_in_m = np.append(pi_over_4_limits_in_m, piOver4LimitIn_m)


                # NOTE: forgot what I was writing in the below comment lol
                # -> Later, will have to add a way to a...

            random_mags_dset[0, idx] = mags
            random_angs_dset[0, idx] = angs

            magnitude_error_unknownUnit_dset[0, idx] = magErrors
            angle_error_unknownUnit_dset[0, idx] = angErrors
            pi_over_4_limit_in_m_dset[0, idx] = pi_over_4_limits_in_m
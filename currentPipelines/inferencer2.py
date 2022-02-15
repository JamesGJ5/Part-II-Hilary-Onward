# PLAN:

# 1) Inference is done on Ronchigrams and 'predicted' Ronchigrams (calculated from predicted labels) are plotted 
# alongside them:

#   Ideally, want the test images used to be images that the network hasn't yet been shown. I am not so sure if the 
#   nominal test images used in training.py have explicitly been shown to the network such that it is trained to 
#   recognise them specifically (invalidating the idea of using them here), but I could always:

#   -- Simulate new Ronchigrams (each featuring only one aberration and from the same ranges as before)
#   -- If I don't simulate new Ronchigrams, make sure I use the same seed as was used in the relevant training run, to 
#   make sure that random_split splits the RonchigramDataset() used then into the same training, evaluation and test 
#   images as before, so I don't accidentally take a set of test images that features a bunch of training images

#   The main challenge will be figuring out how to plot the images side by side and making sure they are all normalised 
#   properly and that the incorrect images aren't accidentally plotted side-by-side.
#   
#   There will also be a challenge with datatypes, I will have to plan that out carefully
#   
#   Won't need to put the calculated Ronchigrams into a DataLoader, probably, but I could probably save them to a HDF5 
#   file (must do so the same way I save my simulations) and then load as before using a RonchigramDataset object if I 
#   want to via torch.utils.data.DataLoader
#       Actually, since for the above, for the DataLoader, ToTensor() necessitates conversion to uint8 (lest it not 
#       do normalisation properly), I will probably see GitHub for the numpy Ronchigram plotting function that used to 
#       be in DataLoader2.py and use that rather than using torch.utils.data.DataLoader. In that case, probably not 
#       even necessary to use RonchigramDataset or HDF5 for the calculated Ronchigrams
#   
#   So, workflow: load data to be inferred normally, apply testTransform for it like ones used in previous relevant 
#   training run, put it in DataLoader, submit a bunch of indices for DataLoader to use, carry out inference on it, 
#   use the predicted labels to calculate new Numpy Ronchigrams, get the same (using the same indices as before)
#   inferred Ronchigrams from RonchigramDataset in numpy array form (i.e. don't apply any transforms), plot said 
#   Ronchigrams alongisde predicted Ronchigrams via a method laid out in an old GitHub save of DataLoader2.py
#   
#   First, before the above, will have to make sure said numpy plotting function leads to correct normalisation. As can 
#   be seen in GitHub history, this function was called show_data(). It is now back in DataLoader2.py, but will of course 
#   need to be built upon in order to plot multiple Roncigrams etc.
#
#        
# 2) A check if inference correctly recognises trends between Ronchigrams: e.g., if we have 100 Ronchigrams and we 
# increase C12 between each of them, while varying the rest of the aberrations randomly, will inference notice this 
# trend?
#
#   To kill two birds with one stone (pardon the idiom) I shall probably simulate Ronchigrams such that the above is 
#   followed


# CODE:

# Importations

import os
import sys
import torch
import model1
from ignite.metrics import MeanAbsoluteError, MeanSquaredError
import math
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision.transforms.functional as F2
from ignite.utils import convert_tensor
import cmath
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision.utils as vutils

# Seed information (may not use the same test set as in training but might as well set the torch seed to be 17 anyway, 
# just in case--I don't see how it can hurt)

fixedSeed = 17

torchSeed = fixedSeed
torch.manual_seed(torchSeed)

# Navigating to the correct working directory

os.chdir("/home/james/VSCode/currentPipelines")
print(f"Current working directory: {os.getcwd()}")

# Adding ../DataLoading to PATH (I think) for RonchigramDataset() importation from DataLoader2.py as well as another function

sys.path.insert(1, "/home/james/VSCode/DataLoading")
from DataLoader2 import RonchigramDataset, showBatch

# Adding ../Simulations to PATH (I think) for importation of calc_ronchigram from Primary_Simulations_1.py

sys.path.insert(2, "/home/james/VSCode/Simulations")
from Primary_Simulation_1 import calc_Ronchigram

# Device configuration (hopefully I will be able to use CPU), think the GPU variable just needs to have a value of "cpu"

GPU = 1
usingGPU = False

if not usingGPU:
    os.environ["CUDA_VISIBLE_DEVICES"]=""

device = torch.device(f"cuda:{GPU}" if usingGPU else "cpu")
print(f"Device being used by PyTorch (non-CUDA): {device}")

if usingGPU:
    torch.cuda.set_device(GPU if usingGPU else "cpu")
    print(f"torch cuda current device: {torch.cuda.current_device()}")

# Model instantiation

efficientNetModel = "EfficientNet-B3"

if efficientNetModel == "EfficientNet-B3":
    parameters = {"num_labels": 8, "width_coefficient": 1.2, "depth_coefficient": 1.4, "dropout_rate": 0.3}
    resolution = 300

model = model1.EfficientNet(num_labels=parameters["num_labels"], width_coefficient=parameters["width_coefficient"], 
                            depth_coefficient=parameters["depth_coefficient"], 
                            dropout_rate=parameters["dropout_rate"]).to(device)

# Loading weights

modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220208-161939/efficientNetBestReciprocalMSE_165727167543.3294"
model.load_state_dict(torch.load(modelPath, map_location = torch.device('cpu')))

# Load RonchigramDataset object with filename equal to the file holding new simulations to be inferred

testSet = RonchigramDataset("/media/rob/hdd2/james/simulations/15_02_22/Single_Aberrations.h5")

# Set up the test transform; it should be the same as testTransform in training.py (1:48pm 15/02/22), with resolution 
# of 300 (as is necessary for EfficientNet-B3) for Resize, along with the same mean and std estimated for the training 
# run in question (it was on 2022 - 02 - 08 (8th February), so see MeanStdLog.txt for that date)
#   Okay, the natural worry is that the mean and std estimated for the data used in the training run won't apply 
#   exactly to the new simulations, but that is probably not too bad since we aren't looking for maximised predictive 
#   performance here, more just correct prediction of trends
# Apply the transform to the RonchigramDataset object too

mean = 0.500990092754364
std = 0.2557201385498047

testTransform = Compose([
    ToTensor(),
    Resize(resolution, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[mean], std=[std])
])

testSet.transform = testTransform

# Batch size and number of workers used to load the data

batchSize = 4
numWorkers = 2

# Collecting subset of testSet to make pretty pictures with

chosenIndices = [200, 404, 551, 805]
testSubset = Subset(testSet, chosenIndices)

# Put set to be inferred into DataLoader along with test transform and choose indices to be used from DataLoader (see 
# oldPipelines/ for how this was done previously)

testLoader = DataLoader(testSubset, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

# Quick tests ot batched data

batch = next(iter(testLoader))
print(f"Size of Ronchigram batch: {batch[0].size()}")
print(f"Size of labels batch: {batch[1].size()}")

testingDataLoader = True

if testingDataLoader:
    for iBatch, batchedSample in enumerate(testLoader):
        print(iBatch, batchedSample[0].size(),
                batchedSample[1].size())

        if iBatch == 0:
            plt.figure()
            showBatch(batchedSample)
            # print(batchedSample["aberrations"])
            plt.ioff()
            plt.show()
            break

# Carry out inference to get predicted labels

model.eval()

with torch.no_grad():
    x = batch

    # yPred is the batch of labels predicted for x
    yPred = model(x[0])

    print("Predicted batch of labels (actualy batch of labels is printed above)\n")
    print(yPred)


# Use predicted labels to calculate new Numpy Ronchigrams (with resolution 1024)

imdim = 1024    # Output Ronchigram will have an array size of imdim x imdim elements
simdim = 100 * 10**-3   # Convergence semi-angle/rad

labelVector = yPred[0]

C10_mag, C10_ang = cmath.polar(labelVector[0] + labelVector[4] * 1j)
C12_mag, C12_ang = cmath.polar(labelVector[1] + labelVector[5] * 1j)
C21_mag, C21_ang = cmath.polar(labelVector[2] + labelVector[6] * 1j)
C23_mag, C23_ang = cmath.polar(labelVector[3] + labelVector[7] * 1j)




# Get the same indices as before and plot the RonchigramDataset Ronchigrams in numpy form (i.e. without transforms)

# Plot calculated Ronchigrams alongside latest ones from RonchigramDataset, using a function like show_data in 
# DataLoader2.py for inspiration

# The above, but for the trend stuff
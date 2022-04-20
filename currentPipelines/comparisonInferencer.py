import os
import sys
import torch
import model1
from ignite.metrics import MeanAbsoluteError, MeanSquaredError
import math
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision.transforms.functional as F2
from ignite.utils import convert_tensor
import cmath
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision.utils as vutils
import datetime
from ignite.utils import convert_tensor
from configparser import ConfigParser
from datetime import date


# Seed information (may not use the same test set as in training but might as well set the torch seed to be 17 anyway, 
# just in case--I don't see how it can hurt)

fixedSeed = 17

torchSeed = fixedSeed
torch.manual_seed(torchSeed)


# Date & time

startTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


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


# OPTIONS

efficientNetModel = "EfficientNet-B2"


# Choosing which labels are going to be returned alongside the Ronchigrams returned by the RonchigramDataset object that 
# shall be instantiated.
chosenVals = {"c10": False, "c12": True, "c21": False, "c23": False, "c30": False,
"c32": False, "c34": False, "c41": False, "c43": False, "c45": False, "c50": False, "c52": False, "c54": False, "c56": False,

"phi10": False, "phi12": True, "phi21": False, "phi23": False, "phi30": False,
"phi32": False, "phi34": False, "phi41": False, "phi43": False, "phi45": False, "phi50": False, "phi52": False, "phi54": False, "phi56": False
}

scalingVals = {
    "c10scaling": 1 / (100 * 10**-9), "c12scaling": 1 / (100 * 10**-9), "c21scaling": 1 / (300 * 10**-9), "c23scaling": 1 / (100 * 10**-9), 
    "c30scaling": 1 / (10.4 * 10**-6), "c32scaling": 1 / (10.4 * 10**-6), "c34scaling": 1 / (5.22 * 10**-6), "c41scaling": 1 / (0.1 * 10**-3), "c43scaling": 1 / (0.1 * 10**-3), "c45scaling": 1 / (0.1 * 10**-3),
    "c50scaling": 1 / (10 * 10**-3), "c52scaling": 1 / (10 * 10**-3), "c54scaling": 1 / (10 * 10**-3), "c56scaling": 1 / (10 * 10**-3),

    "phi10scaling": 1, "phi12scaling": 1 / (2 * np.pi / 2), "phi21scaling": 1 / (2 * np.pi / 1), "phi23scaling": 1 / (2 * np.pi / 3), 
    "phi30scaling": 1, "phi32scaling": 1 / (2 * np.pi / 2), "phi34scaling": 1 / (2 * np.pi / 4), "phi41scaling": 1 / (2 * np.pi / 1), "phi43scaling": 1 / (2 * np.pi / 3), "phi45scaling": 1 / (2 * np.pi / 5),
    "phi50scaling": 1, "phi52scaling": 1 / (2 * np.pi / 2), "phi54scaling": 1 / (2 * np.pi / 4), "phi56scaling": 1 / (2 * np.pi / 6)
} 


# NUMBER OF LABELS FOR MODEL TO PREDICT

# numLabels is essentially the number of elements the model outputs in its prediction for a given Ronchigram. It is of 
# course best to match this number to the same number that was used in training the model.
numLabels = 2


# CONFIG STUFF

# Here, I am only putting things in config2.ini for which there are many options with a lot of accompanying commentary, 
# e.g. in the case of model paths next to which I state where the model is from.

config = ConfigParser()
config.read("config2.ini")


# MODEL PATH

# This is the path of the model to be used for inference in this script
# NOTE: mean and std are the mean and standard deviation estimated for the data used to train the model whose path 
# is modelPath; can be found in modelLogging

modelPath = config['modelSection']['modelPath']
desiredSimdim = eval(config['modelSection']['desiredSimdim'])
actualSimdim = eval(config['modelSection']['actualSimdim'])


# TEST SET

# The path of the Ronchigrams which are to be inferred and whose "predicted" Ronchigrams are to be plotted alongside 
# them.

testSetPath = config["testSetPath"]["testSetPath"]
testSetMean = eval(config["testSetPath"]["mean"])
testSetStd = eval(config["testSetPath"]["std"])


# SCALING TENSORS
# Just initialising some torch Tensors that will be useful for below calculations
# TODO: find a way to do the below more efficiently

# sorted(chosenVals) sorts the keys in chosenVals, chosenVals[key] extracts whether its value is True or False; so, sortedChosenVals is a 
# list whose elements are either True or False, corresponding to whether the corresponding aberration constants key in sorted(chosenVals) is 
# included or not in scaling predicted and target labels to real values for Ronchigram depiction etc.
sortedChosenVals = [chosenVals[key] for key in sorted(chosenVals)]

# The below returns the scaling factors corresponding to the aberrations being represented by "True"
usedScalingFactors = [scalingVals[sorted(scalingVals)[i]] for i, x in enumerate(sortedChosenVals) if x]

usedScalingFactors = torch.tensor(usedScalingFactors)


# MODEL INSTANTIATION

if efficientNetModel == "EfficientNet-B3":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.2, "depth_coefficient": 1.4, "dropout_rate": 0.3}
    resolution = 300

elif efficientNetModel == "EfficientNet-B2":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.1, "depth_coefficient": 1.2, "dropout_rate": 0.3}
    resolution = 260

model = model1.EfficientNet(num_labels=parameters["num_labels"], width_coefficient=parameters["width_coefficient"], 
                            depth_coefficient=parameters["depth_coefficient"], 
                            dropout_rate=parameters["dropout_rate"]).to(device)


# LOADING WEIGHTS

model.load_state_dict(torch.load(modelPath, map_location = torch.device(f"cuda:{GPU}" if usingGPU else "cpu"))["model"])


# TEST DATA IMPORTATION
# Load RonchigramDataset object with filename equal to the file holding new simulations to be inferred

testSet = RonchigramDataset(testSetPath, complexLabels=False, **chosenVals, **scalingVals)

apertureSize = testSet[0][0].shape[0] / 2 * desiredSimdim / actualSimdim

testTransform = Compose([
    ToTensor(),
    CenterCrop(np.sqrt(2) * apertureSize),
    Resize(resolution, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[testSetMean], std=[testSetStd])
])

testSet.transform = testTransform


# RONCHIGRAMS TO MAKE PRETTY PICTURES WITH

chosenIndices = [0, round(len(testSet) / 4), round(len(testSet) / 2), round(len(testSet) * 3 / 4)]
testSubset = Subset(testSet, chosenIndices)


# DATA LOADING

batchSize = 32
numWorkers = 8

testLoader = DataLoader(testSubset, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

# Quick tests to see if data is batched correctly
batch = next(iter(testLoader))

testingDataLoader = False

if testingDataLoader:
    for iBatch, batchedSample in enumerate(testLoader):

        print(f"\nBatch index: {iBatch}")
        print(f"Ronchigram batch size: {batchedSample[0].size()}")
        print(f"Labels batch size: {batchedSample[1].size()}\n")

        if iBatch == 0:
            plt.figure()

            # In batchedSample, labels get printed, hence the below print statement
            print("Batch of target labels:")
            showBatch(batchedSample)

            plt.ioff()
            plt.show()

            break


# INFERENCE TO GET PREDICTED LABELS

model.eval()

with torch.no_grad():
    # NOTE: if this isn't feasible GPU memory-wise, may want to replace batch with batch[0] and instances of x[0] with x
    x = convert_tensor(batch[0], device=device, non_blocking=True)

    # yPred is the batch of labels predicted for x
    yPred = model(x)

    # print("\nBatch of predicted labels:")
    # print(yPred)

    # The below is done because before input, cnm and phinm values are scaled by scaling factors; to see what predictions 
    # mean physically, must rescale back as is done below.
    yPred = yPred.cpu() / usedScalingFactors

print("\nBatch of target labels but un-normalised:")
print(batch[1] / usedScalingFactors)

print("\nBatch of predicted labels but un-normalised:")
print(yPred)


# CALCULATING PREDICTED RONCHIGRAMS

imdim = 1024    # Output Ronchigram will have an array size of imdim x imdim elements

# TODO: make simdim importable alongside the simulations path that is imported
simdim = 70 * 10**-3   # Convergence semi-angle/rad

# NOTE: this will contain numpy arrays, not torch Tensors
predictedRonchBatch = np.empty((batchSize, imdim, imdim, 1))
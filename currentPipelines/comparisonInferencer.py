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
from numpy.random import default_rng
from matplotlib_scalebar.scalebar import ScaleBar
from textwrap import wrap


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

GPU = 0
usingGPU = True

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
chosenVals = {
    "c10": True, "c12": True, "c21": True, "c23": True, "c30": True,
    "c32": True, "c34": True, "c41": True, "c43": True, "c45": True,
    "c50": True, "c52": True, "c54": True, "c56": True,

    "phi10": True, "phi12": True, "phi21": True, "phi23": True, "phi30": True,
    "phi32": True, "phi34": True, "phi41": True, "phi43": True, "phi45": True,
    "phi50": True, "phi52": True, "phi54": True, "phi56": True
}

c12scaling = 1 / (2.449 * 10**-9)
phi12scaling = 1 / (2 * np.pi / 2)

scalingVals = {
    "c10scaling": 1, "c12scaling": c12scaling, "c21scaling": 1, "c23scaling": 1, 
    "c30scaling": 1, "c32scaling": 1, "c34scaling": 1, "c41scaling": 1, "c43scaling": 1, "c45scaling": 1,
    "c50scaling": 1, "c52scaling": 1, "c54scaling": 1, "c56scaling": 1,

    "phi10scaling": 1, "phi12scaling": phi12scaling, "phi21scaling": 1, "phi23scaling": 1, 
    "phi30scaling": 1, "phi32scaling": 1, "phi34scaling": 1, "phi41scaling": 1, "phi43scaling": 1, "phi45scaling": 1,
    "phi50scaling": 1, "phi52scaling": 1, "phi54scaling": 1, "phi56scaling": 1
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
seed = fixedSeed

chosenIndices = default_rng(seed).integers(low=0, high=len(testSet), size=4)
testSubset = Subset(testSet, chosenIndices)


# DATA LOADING
batchSize = len(chosenIndices)
numWorkers = 8

testLoader = DataLoader(testSubset, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

# For quick tests to see if data is batched correctly
batch = next(iter(testLoader))

ronchBatch = batch[0]
labelsBatch = batch[1]

# These indices only work when the labels are c10, c12, c21, c23, c30, c32, c34, c41, c43, c45, c50, c52, c54, c56, 
# phi10, phi12, phi21, phi23, phi30, phi32, phi34, phi41, phi43, phi45, phi50, phi52, phi54, phi56
if len(usedScalingFactors) == 28:
    c12index = 1
    phi12index = 15

else:
    c12index = input("Index of c12 in a label vector: ")
    phi12index = input("Index of phi12 in a label vector: ")

# For some post-inference work
c12batch = torch.reshape(labelsBatch[:, c12index], (labelsBatch.size(dim=0), 1))
phi12batch = torch.reshape(labelsBatch[:, phi12index], (labelsBatch.size(dim=0), 1))

c12phi12batch = torch.cat((c12batch, phi12batch), 1)

c12phi12scalingFactors = torch.tensor([usedScalingFactors[c12index]] + [usedScalingFactors[phi12index]])

# New batch with the same form as the old batch except with the labels only containing values for c12 & phi12
batch = [ronchBatch, c12phi12batch]

testingDataLoader = False
if testingDataLoader:
    for iBatch, batchedSample in enumerate([batch]):

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
    yPred = yPred.cpu() / c12phi12scalingFactors

# Target c12 & phi12
print("\nBatch of target c12 & phi12 but un-normalised:")
print(c12phi12batch / c12phi12scalingFactors)

# Predicted c12 & phi12
print("\nBatch of predicted c12 & phi12 but un-normalised:")
print(yPred)


# CALCULATING PREDICTED RONCHIGRAMS

imdim = 1024    # Output Ronchigram will have an array size of imdim x imdim elements

# TODO: make simdim importable alongside the simulations path that is imported
simdim = 180 * 10**-3   # Convergence semi-angle/rad

# NOTE: this will contain numpy arrays, not torch Tensors
predictedRonchBatch = np.empty((batchSize, imdim, imdim, 1))

for labelVectorIndex in range(batchSize):

    labelVector = yPred[labelVectorIndex]

    C12_mag, C12_ang = [element.item() for element in labelVector]

    other_mags = [labelsBatch[labelVectorIndex][i].item() for i in [0] + list(range(2, 14))]
    other_angs = [labelsBatch[labelVectorIndex][i].item() for i in [14] + list(range(16, 28))]

    # Putting c12 and phi12 together with other aberration constants of the same type
    other_mags.insert(1, C12_mag)
    mag_list = other_mags

    other_angs.insert(1, C12_ang)
    ang_list = other_angs

    I, t, seed = testSet.get_I_t_Seed(idx=chosenIndices[labelVectorIndex])
    print(I, t, seed)
    b = 1

    predictedRonch = calc_Ronchigram(imdim, simdim, *mag_list, *ang_list, I=I, b=b, t=t, aperture_size=simdim, 
                                        zhiyuanRange=False, seed=seed)

    # For the colour channel position in this numpy array
    predictedRonch = np.expand_dims(predictedRonch, 2)

    predictedRonchBatch[labelVectorIndex] = predictedRonch

# NOTE: IF YOU SEE A BUNCH OF ZEROS WHEN YOU PRINT THE BELOW BUT THE PREDICTED RONCHIGRAM HAS NONZERO 2-FOLD 
# ASTIGMATISM MAGNITUDE AND ANGLE, IT MAY JUST BE PLOTTING THE BITS MASKED BY THE OBJECTIVE APERTURE
print(predictedRonchBatch[0])
print(predictedRonchBatch[0].shape)

# Retrieving the data inferred by the network in numpy array form this time (i.e. without transforms)
testSet.transform = None
testSubset = Subset(testSet, chosenIndices)


# Plot calculated Ronchigrams alongside latest ones from RonchigramDataset, using a function like show_data in 
# DataLoader2.py for inspiration

scale = 2*simdim/imdim * 1000    # mrad per pixel
# scalebar = ScaleBar(scale, units="mrad", dimension="angle")

for i in range(len(testSubset)):
# for i in [0]:
    ax = plt.subplot(1, 2, 1)

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    ax.imshow(testSubset[i][0], cmap="gray")

    scalebar = ScaleBar(scale, units="mrad", dimension="angle")
    ax.add_artist(ScaleBar(scale, units="mrad", dimension="angle"))

    aberConstants = testSubset[i][1]

    l1 = []

    for j, const in enumerate(aberConstants):

        if j <= 13:

            if j in [0, 2, 3]:

                # print(const)
                const /= 10**-9

            elif j == 1:

                const /= c12scaling
                const /= 10**-9

            elif j in [4, 5, 6]:

                const /= 10**-6

            else:

                const /= 10**-3
        
        if j >= 14:

            if j in [14, 18, 24]:

                continue

            elif j == 15:

                const /= phi12scaling
                const /= np.pi

            else:
                
                const /= np.pi

        const = const.item()
        const = round(const, 2)

        l1.append(const)

    # aberConstantsString = f'c1,0={l1[0]}nm, \33[1mc1,2={l1[1]}nm' + f'\33[m, c2,1={l1[2]}nm, c2,3={l1[3]}nm, c3,0={l1[4]}um, c3,2={l1[5]}um, c3,4={l1[6]}um, c4,1={l1[7]}mm, c4,3={l1[8]}mm, c4,5={l1[9]}mm, c5,0={l1[10]}mm, c5,2={l1[11]}mm, c5,4={l1[12]}mm, c5,6={l1[13]}mm, \33[1m\u03A61,2={l1[14]}\u03C0' + f'\33[m, \u03A62,1={l1[15]}\u03C0, \u03A62,3={l1[16]}\u03C0, \u03A63,2={l1[17]}\u03C0, \u03A63,4={l1[18]}\u03C0, \u03A64,1={l1[19]}\u03C0, \u03A64,3={l1[20]}\u03C0, \u03A64,5={l1[21]}\u03C0, \u03A65,2={l1[22]}\u03C0, \u03A65,4={l1[23]}\u03C0, \u03A65,6={l1[24]}\u03C0'
    aberConstantsString = f'c1,0={l1[0]}nm, c1,2={l1[1]}nm, c2,1={l1[2]}nm, c2,3={l1[3]}nm, c3,0={l1[4]}um, c3,2={l1[5]}um, c3,4={l1[6]}um, c4,1={l1[7]}mm, c4,3={l1[8]}mm, c4,5={l1[9]}mm, c5,0={l1[10]}mm, c5,2={l1[11]}mm, c5,4={l1[12]}mm, c5,6={l1[13]}mm, \u03A61,2={l1[14]}\u03C0, \u03A62,1={l1[15]}\u03C0, \u03A62,3={l1[16]}\u03C0, \u03A63,2={l1[17]}\u03C0, \u03A63,4={l1[18]}\u03C0, \u03A64,1={l1[19]}\u03C0, \u03A64,3={l1[20]}\u03C0, \u03A64,5={l1[21]}\u03C0, \u03A65,2={l1[22]}\u03C0, \u03A65,4={l1[23]}\u03C0, \u03A65,6={l1[24]}\u03C0'

    # print(aberConstantsString)

    ax.title.set_text('\n'.join(wrap(aberConstantsString, width=60)))


    ax = plt.subplot(1, 2, 2)

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    ax.imshow(predictedRonchBatch[i], cmap="gray")

    scalebar = ScaleBar(scale, units="mrad", dimension="angle")
    ax.add_artist(ScaleBar(scale, units="mrad", dimension="angle"))

    c12Pred = (yPred[i][0]).item()
    c12Pred /= 10**-9
    c12Pred = round(c12Pred, 2)

    phi12Pred = (yPred[i][1]).item()
    phi12Pred /= np.pi
    phi12Pred = round(phi12Pred, 2)

    predConstsString = f'c1,2={c12Pred}nm, \u03A61,2={phi12Pred}\u03C0'

    ax.title.set_text('\n'.join(wrap(predConstsString, width=60)))

    plt.subplots_adjust(wspace=0, hspace=0, top=0.8)

    plt.show()
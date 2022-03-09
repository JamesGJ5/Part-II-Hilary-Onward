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
import datetime
from ignite.utils import convert_tensor


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
usingGPU = True

if not usingGPU:
    os.environ["CUDA_VISIBLE_DEVICES"]=""

device = torch.device(f"cuda:{GPU}" if usingGPU else "cpu")
print(f"Device being used by PyTorch (non-CUDA): {device}")

if usingGPU:
    torch.cuda.set_device(GPU if usingGPU else "cpu")
    print(f"torch cuda current device: {torch.cuda.current_device()}")


# OPTIONS

efficientNetModel = "EfficientNet-B3"

predictSingleAber = True   # I.E. whether only a single aberration is to have its constants predicted from a test aberration

if predictSingleAber:
    singleAber = "C10"

mixedRonchs = True  # Whether or not the Ronchigrams being inferred from contain more than one aberration

# Choosing which labels are going to be returned alongside the Ronchigrams returned by the RonchigramDataset object that 
# shall be instantiated.
chosenVals = {"c10": True, "c12": False, "c21": False, "c23": False, "phi10": False, "phi12": False, "phi21": False, "phi23": False}
scalingVals = {
    "c10scaling": 10**7, "c12scaling": 10**7, "c21scaling": 10**5, "c23scaling": 10**5, 
    "phi10scaling": 1, "phi12scaling": 1 / (np.pi / 2), "phi21scaling": 1 / (np.pi), "phi23scaling": 1 / (np.pi / 3)
} 


# NUMBER OF LABELS FOR MODEL TO PREDICT

# numLabels is essentially the number of elements the model outputs in its prediction for a given Ronchigram. It is of 
# course best to match this number to the same number that was used in training the model.
numLabels = 1


# MODEL PATH

# This is the path of the model to be used for inference in this script
# NOTE: mean and std are the mean and standard deviation estimated for the data used to train the model whose path 
# is modelPath; can be found in modelLogging

# This path is for a model trained to recognise JUST c10 on MIXED-ABERRATION Ronchigrams with max_c10 100nm, max_c12 50nm,
# max_c21 5000nm, max_c23 5000nm, phi10 0, phi12 0 to pi/2, phi21 0 to pi, and phi23 0 to pi/3.
modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220307-205356/best_model_Loss=0.1338.pt"
mean = 0.5011
std = 0.2492

# This path is for a model trained to recognise JUST c10 on MIXED-ABERRATION Ronchigrams with max_c10 100nm, max_c12 100nm,
# max_c21 10000nm, max_c23 10000nm, phi10 0, phi12 0 to pi/2, phi21 0 to pi, and phi23 0 to pi/3.
# modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220305-124423/best_model_Loss=0.1107.pt"
# mean = 0.5009
# std = 0.2483

# This path is for a model trained to recognise c10, c12, c21, c23, phi12, phi21, phi23 on MIXED-ABERRATION Ronchigrams 
# with max_c10 100nm, max_c12 100nm, max_c21 10000nm, max_c23 10000nm, phi10 0, phi12 0 to pi/2, phi21 0 to pi, and phi23 0 to pi/3.
# modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220306-164551/best_model_Loss=0.0867.pt"
# mean = 0.5009
# std = 0.2483

# Model trained on approx. 100,000 Ronchigrams containing JUST C12
# modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220226-220806/best_model_Loss=0.1902.pt"

# Model trained on approx. 100,000 Ronchigrams containing JUST C21
# modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220227-112003/best_model_Loss=0.0885.pt"

# Model trained on approx. 100,000 Ronchigrams containing JUST C23
# modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220228-003811/best_model_Loss=0.1071.pt"


# TEST SET PATH

# The path of the Ronchigrams which are to be inferred and whose "predicted" Ronchigrams are to be plotted alongside 
# them.

# 4 Ronchigrams containing only c10 (0, 25nm, 50nm, 75nm), c21 (fixed at 1000nm) and phi21 (fixed at 0)
testSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/04_03_22/c10_0_to_75nm_c21_1000nm.h5"

# # Approx. 1000 Ronchigrams containing only c12 (from 0 to 100nm) and phi12 (from 0 to pi/2 rad)
# testSetPath = "/media/rob/hdd1/james-gj/Simulations/25_02_22/Single_C12.h5"

# # Approx. 1000 Ronchigrams containing only c21 (from 0 to 10000nm) and phi21 (from 0 to pi rad)
# testSetPath = "/media/rob/hdd1/james-gj/Simulations/26_02_22/Single_C21.h5"

# # Approx. 1000 Ronchigrams containing only c23 (from 0 to 10000nm) and phi23 (from 0 to pi/3 rad)
# testSetPath = "/media/rob/hdd1/james-gj/Simulations/26_02_22/Single_C23.h5"


# TREND SET PATH

# The path of Ronchigrams in which there's a trend that I want to see if the model can predict

# This is for models at 20220307-205356/
trendSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/08_03_22/linearC10.h5"

# This is for models at 20220305-124423/ and 20220306-164551/
# trendSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/Linear_C10.h5"

# Approx. 1000 Ronchigrams in which c12 varies linearly from 0 to 100nm and other constants are random between 0 and 
# 100nm for c10, 0 and 10000nm for c21 and c23, 0 and pi/2 rad for c12, and 0 and pi/3 rad for c23.
# trendSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/27_02_22/Linear_C12.h5"

# Approx. 1000 Ronchigrams in which c21 varies linearly from 0 to 10000nm and other constants are random between 0 and 
# 100nm for c10 and c12, 0 and 10000nm for c23, 0 and pi/2 rad for c12, and 0 and pi/3 rad for c23.
# trendSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/27_02_22/Linear_C21.h5"

# Approx. 1000 Ronchigrams in which c23 varies linearly from 0 to 10000nm and other constants are random between 0 and 
# 100nm for c10 and c12, 0 and 10000nm for c21, 0 and pi/2 rad for c12, and 0 and pi/3 rad for c23.
# trendSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/27_02_22/Linear_C23.h5"


# SCALING TENSORS
# Just initialising some torch Tensors that will be useful for below calculations
# TODO: find a way to do the below more efficiently

# sorted(chosenVals) sorts the keys in chosenVals, chosenVals[key] extracts whether its value is True or False; so, sortedChosenVals is a 
# list whose elements are either True or False, corresponding to whether the corresponding aberration constatns key in sorted(chosenVals) is 
# included or not in scaling predicted and target labels to real values for Ronchigram depiction etc.
sortedChosenVals = [chosenVals[key] for key in sorted(chosenVals)]

# The below returns the scaling factors corresponding to the aberrations being represented by "True"
usedScalingFactors = [scalingVals[sorted(scalingVals)[i]] for i, x in enumerate(sortedChosenVals) if x]

usedScalingFactors = torch.tensor(usedScalingFactors)


# MODEL INSTANTIATION

if efficientNetModel == "EfficientNet-B3":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.2, "depth_coefficient": 1.4, "dropout_rate": 0.3}
    resolution = 300

model = model1.EfficientNet(num_labels=parameters["num_labels"], width_coefficient=parameters["width_coefficient"], 
                            depth_coefficient=parameters["depth_coefficient"], 
                            dropout_rate=parameters["dropout_rate"]).to(device)


# LOADING WEIGHTS

model.load_state_dict(torch.load(modelPath, map_location = torch.device(f"cuda:{GPU}" if usingGPU else "cpu")))


# TEST DATA IMPORTATION
# Load RonchigramDataset object with filename equal to the file holding new simulations to be inferred

testSet = RonchigramDataset(testSetPath, complexLabels=False, **chosenVals, **scalingVals)


# Set up the test transform; it should be the same as testTransform in training.py (1:48pm 15/02/22), with resolution 
# of 300 (as is necessary for EfficientNet-B3) for Resize, along with the same mean and std estimated for the training 
# run in question (it was on 2022 - 02 - 08 (8th February), so see MeanStdLog.txt for that date)
#   Okay, the natural worry is that the mean and std estimated for the data used in the training run won't apply 
#   exactly to the new simulations, but that is probably not too bad since we aren't looking for maximised predictive 
#   performance here, more just correct prediction of trends
# Apply the transform to the RonchigramDataset object too

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

chosenIndices = [0, 1, 2, 3]
testSubset = Subset(testSet, chosenIndices)


# Put set to be inferred into DataLoader along with test transform and choose indices to be used from DataLoader (see 
# oldPipelines/ for how this was done previously)

testLoader = DataLoader(testSubset, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)


# Quick tests on batched data

batch = next(iter(testLoader))
print(f"Size of Ronchigram batch: {batch[0].size()}")
print(f"Size of labels batch: {batch[1].size()}")
print(batch[1])

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
    # NOTE: if this isn't feasible GPU memory-wise, may want to replace batch with batch[0] and instances of x[0] with x
    x = convert_tensor(batch, device=device, non_blocking=True)

    # yPred is the batch of labels predicted for x
    yPred = model(x[0])

    print("Predicted batch of labels (batch of actual labels is printed above)\n")
    print(yPred)

    # The below is done because for training, cnm and phinm values are scaled by scaling factors; to see what predictions 
    # mean physically, must rescale back
    # TODO: generalise the below to other scaling values
    yPred = yPred.cpu() / usedScalingFactors
    print(yPred)


# Use predicted labels to calculate new Numpy Ronchigrams (with resolution 1024)

imdim = 1024    # Output Ronchigram will have an array size of imdim x imdim elements
simdim = 100 * 10**-3   # Convergence semi-angle/rad

# NOTE: this will contain numpy arrays, not torch Tensors
predictedRonchBatch = np.empty((batchSize, imdim, imdim, 1))

for labelVectorIndex in range(batchSize):
    labelVector = yPred[labelVectorIndex]

    # If the network was trained using complex labels, the predicted labels must contain predicted real & imaginary 
    # parts of complex forms of aberrations
    if testSet.complexLabels:

        C10_mag, C10_ang = cmath.polar(labelVector[0] + labelVector[4] * 1j)
        C12_mag, C12_ang = cmath.polar(labelVector[1] + labelVector[5] * 1j)
        C21_mag, C21_ang = cmath.polar(labelVector[2] + labelVector[6] * 1j)
        C23_mag, C23_ang = cmath.polar(labelVector[3] + labelVector[7] * 1j)

    # TODO: replace the below with a concise generator or something
    # TODO: remember that just because an aberration magnitude is zero doesn't necessarily mean angle is zero? Idk
    i = 0

    C10_mag = labelVector[i].item() if chosenVals["c10"] else 0
    i = i + 1 if C10_mag != 0 else i

    C12_mag = labelVector[i].item() if chosenVals["c12"] else 0
    i = i + 1 if C12_mag != 0 else i

    C21_mag = labelVector[i].item() if chosenVals["c21"] else 0
    i = i + 1 if C21_mag != 0 else i

    C23_mag = labelVector[i].item() if chosenVals["c23"] else 0
    i = i + 1 if C23_mag != 0 else i

    C10_ang = labelVector[i].item() if chosenVals["phi10"] else 0
    i = i + 1 if C10_ang != 0 else i

    C12_ang = labelVector[i].item() if chosenVals["phi12"] else 0
    i = i + 1 if C12_ang != 0 else i

    C21_ang = labelVector[i].item() if chosenVals["phi21"] else 0
    i = i + 1 if C21_ang != 0 else i

    C23_ang = labelVector[i].item() if chosenVals["phi23"] else 0
    i = i + 1 if C23_ang != 0 else i

    print(C10_mag, C12_mag, C21_mag, C23_mag, C10_ang, C12_ang, C21_ang, C23_ang)

    I, t = testSet.getIt(chosenIndices[labelVectorIndex])
    print(I, t)

    # NOTE: for now, haven't been saving b, it will probably just remain as 1 but I may change it so be careful
    b = 1

    # TODO: calculate Ronchigram here from parameters above
    predictedRonch = calc_Ronchigram(imdim=imdim, simdim=simdim,
                                    C10_mag=C10_mag, C12_mag=C12_mag, C21_mag=C21_mag, C23_mag=C23_mag,
                                    C10_ang=C10_ang, C12_ang=C12_ang, C21_ang=C21_ang, C23_ang=C23_ang,
                                    I=I, b=b, t=t)

    predictedRonch = np.expand_dims(predictedRonch, 2)

    # print(predictedRonch[0].shape)
    # print(predictedRonch)

    predictedRonchBatch[labelVectorIndex] = predictedRonch

# print(predictedRonchBatch)
# print(predictedRonchBatch[0].shape)


# Retrieving the data inferred by the network in Numpy form this time (i.e. without transforms)

testSet.transform = None
testSubset = Subset(testSet, chosenIndices)

# print(type(testSubset[0][0]))
# print(testSubset[0][0].shape)
# print(testSubset[0][0])

# Plot calculated Ronchigrams alongside latest ones from RonchigramDataset, using a function like show_data in 
# DataLoader2.py for inspiration

for i in range(len(testSubset)):
    plt.subplot(2, len(testSubset), i + 1)
    plt.imshow(testSubset[i][0], cmap="gray")

    plt.subplot(2, len(testSubset), i + 5)
    plt.imshow(predictedRonchBatch[i], cmap="gray")

plt.show()


# Checking trends

# PLAN:
#
# The model weights I use for this part will probably be the same as are used for the previous part
# REMEMBER THAT SCALING FACTORS ARE A THING, probably best to divide actual and predicted labels by scaling factor just 
# so the plots are more clear regarding what they represent
#
# 1) Instantiate a RonchigramDataset object from the file Linear_C10.h5
# 2) Use the same testTransform as before, with the same mean and std (I.E. those calculated for the training data), so 
# set RonchigramDataset object's transform attribute to testTransform
# 3) Create torch.utils.data.DataLoader using the RonchigramDataset object with shuffle == False
# 4) Using Kaggle for inspiration, make model predict for each Ronchigram, appending each predicted c10 to a numpy array 
# with 1 dimension; also, append each actual c10 to a numpy array in the process
# 5) Plot actual c10 array vs predicted c10 array on the same graph, in different colours/maybe one with a line and one 
# as scattered dots (the predictions being the dots)

trendSet = RonchigramDataset(trendSetPath, transform=testTransform, complexLabels=False, **chosenVals, **scalingVals)

trendLoader = DataLoader(trendSet, batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True, 
                        drop_last=False)

# print(trendSet[0][1].size())

# targetArray = np.empty((len(trendSet), *trendSet[0][1].size()))
# print(targetArray.shape)

targetTensor = torch.tensor([])
predTensor = torch.tensor([])

with torch.no_grad():
    for batchIdx, (batchedRonchs, batchedTargets) in enumerate(trendLoader):

        batchedRonchs = convert_tensor(batchedRonchs, device=device, non_blocking=True)

        # TODO: change below so that instead of flattening, it reshapes into a different row for each cnm and phinm
        batchedTargets = batchedTargets[:, 0].cpu()
        predBatch = model(batchedRonchs)[:, 0].cpu()

        targetTensor = torch.cat((targetTensor, batchedTargets))
        predTensor = torch.cat((predTensor, predBatch))

        if batchIdx % 10 == 0: print(f"{batchIdx} batches done...")

    # usedScalingFactors = torch.reshape(usedScalingFactors, (len(usedScalingFactors), 1))

    # TODO: generalise the below to other scaling values
    targetTensor = (targetTensor / usedScalingFactors[0]).numpy()
    predTensor = (predTensor / usedScalingFactors[0]).numpy()

    plt.plot(np.linspace(1, len(targetTensor), len(targetTensor)), targetTensor, 'b')
    plt.plot(np.linspace(1, len(predTensor), len(predTensor)), predTensor, 'ro')
    plt.ylabel("blue: target, red: prediction")
    # plt.savefig(f"/media/rob/hdd1/james-gj/Simulations/inferenceResults/trendGraphs/{startTime}.png")
    plt.show()
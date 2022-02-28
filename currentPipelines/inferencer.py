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
#   To kill two birds with one stone (pardon the idiom) I shall probably simulate Ronchigrams that are suitable for both 
#   Steps 1 & 2 above.


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


# Options

efficientNetModel = "EfficientNet-B3"
singleAber = "C10"

chosenVals = {"c10": False, "c12": False, "c21": False, "c23": False, "phi10": False, "phi12": False, "phi21": False, "phi23": False}
scalingVals = {
    "c10scaling": 10**7, "c12scaling": 10**7, "c21scaling": 10**5, "c23scaling": 10**5, 
    "phi10scaling": 1, "phi12scaling": 1 / (np.pi / 2), "phi21scaling": 1 / (np.pi), "phi23scaling": 1 / (np.pi / 3)
}

if singleAber == "C10":

    numLabels = 1
    chosenVals["c10"] = True
    modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220225-174816/best_model_Loss=0.4529.pt"
    testSetPath = "/media/rob/hdd1/james-gj/Simulations/22_02_22/Single_C10.h5"

    # NOTE: mean and std were retrieved from modelLogging, logged when modelPath was created
    mean = 0.5011
    std = 0.2560

    trendSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/Linear_C10.h5"

if singleAber == "C12":

    numLabels = 2
    chosenVals["c12"] = True
    chosenVals["phi12"] = True
    modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220226-220806/best_model_Loss=0.1902.pt"
    testSetPath = "/media/rob/hdd1/james-gj/Simulations/25_02_22/Single_C12.h5"

    # NOTE: mean and std were retrieved from modelLogging, logged when modelPath was created
    mean = 0.5010
    std = 0.2544

    trendSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/Linear_C12.h5"

if singleAber == "C21":

    numLabels = 2
    chosenVals["c21"] = True
    chosenVals["phi21"] = True
    modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220227-112003/best_model_Loss=0.0885.pt"
    testSetPath = "/media/rob/hdd1/james-gj/Simulations/26_02_22/Single_C21.h5"

    # NOTE: mean and std were retrieved from modelLogging, logged when modelPath was created
    mean = 0.5006
    std = 0.2502

    trendSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/Linear_C21.h5"

if singleAber == "C23":

    numLabels = 2
    chosenVals["c23"] = True
    chosenVals["phi23"] = True
    modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220228-003811/best_model_Loss=0.1071.pt"
    testSetPath = "/media/rob/hdd1/james-gj/Simulations/26_02_22/Single_C23.h5"

    # NOTE: mean and std were retrieved from modelLogging, logged when modelPath was created
    mean = 0.5007
    std = 0.2488

    trendSetPath = "/media/rob/hdd1/james-gj/Simulations/forInference/Linear_C23.h5"


# Scaling tensors
# Just initialising some torch Tensors that will be useful for below calculations

sortedChosenVals = [chosenVals[key] for key in sorted(chosenVals)]
usedScalingFactors = [scalingVals[sorted(scalingVals)[i]] for i, x in enumerate(sortedChosenVals) if x]

usedScalingFactors = torch.tensor(usedScalingFactors)

# print(usedScalingFactors)


# Model instantiation

if efficientNetModel == "EfficientNet-B3":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.2, "depth_coefficient": 1.4, "dropout_rate": 0.3}
    resolution = 300

model = model1.EfficientNet(num_labels=parameters["num_labels"], width_coefficient=parameters["width_coefficient"], 
                            depth_coefficient=parameters["depth_coefficient"], 
                            dropout_rate=parameters["dropout_rate"]).to(device)


# Loading weights

model.load_state_dict(torch.load(modelPath, map_location = torch.device(f"cuda:{GPU}" if usingGPU else "cpu")))


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

testingDataLoader = False

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
    yPred /= scalingVals["c10scaling"]


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
    i = 0

    C10_mag = labelVector[i].item() if chosenVals["c10"] else 0
    i = i + 1 if C10_mag != 0 else i

    C12_mag = labelVector[i].item() if chosenVals["c12"] else 0
    i = i + 1 if C10_mag != 0 else i

    C21_mag = labelVector[i].item() if chosenVals["c21"] else 0
    i = i + 1 if C10_mag != 0 else i

    C23_mag = labelVector[i].item() if chosenVals["c23"] else 0
    i = i + 1 if C10_mag != 0 else i

    C10_ang = labelVector[i].item() if chosenVals["phi10"] else 0
    i = i + 1 if C10_mag != 0 else i

    C12_ang = labelVector[i].item() if chosenVals["phi12"] else 0
    i = i + 1 if C10_mag != 0 else i

    C21_ang = labelVector[i].item() if chosenVals["phi21"] else 0
    i = i + 1 if C10_mag != 0 else i

    C23_ang = labelVector[i].item() if chosenVals["phi23"] else 0
    i = i + 1 if C10_mag != 0 else i


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

# sys.exit()


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

    # TODO: generalise the below to other scaling values
    targetTensor = (targetTensor / scalingVals["c10scaling"]).numpy()
    predTensor = (predTensor / scalingVals["c10scaling"]).numpy()

    plt.plot(np.linspace(1, len(targetTensor), len(targetTensor)), targetTensor, 'b')
    plt.plot(np.linspace(1, len(predTensor), len(predTensor)), predTensor, 'ro')
    plt.ylabel("blue: target, red: prediction")
    plt.savefig(f"/media/rob/hdd1/james-gj/Simulations/forInference/trendGraphs/{startTime}.png")
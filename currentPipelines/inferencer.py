# Importations

import os
import sys
import torch
import model1
from ignite.metrics import MeanAbsoluteError, MeanSquaredError
import math
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision.transforms.functional as F2
from ignite.utils import convert_tensor
import cmath
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision.utils as vutils


# Seed information

fixedSeed = 17

# Might make a way for seed to be random later
torchSeed = fixedSeed
torch.manual_seed(torchSeed)


# Navigating to the correct current working directory

os.chdir("/home/james/VSCode/currentPipelines")
print(f"Current working directory: {os.getcwd()}")


# Adding ../DataLoading to PATH (I think) for RonchigramDataset() importation from DataLoader2.py

sys.path.insert(1, "/home/james/VSCode/DataLoading")
from DataLoader2 import RonchigramDataset


# Adding ../Simulations to PATH (I think) for importation of calc_ronchigram from Primary_Simulations_1.py

sys.path.insert(2, "/home/james/VSCode/Simulations")
from Primary_Simulation_1 import calc_Ronchigram


# Device configuration

GPU = 1
device = torch.device(f"cuda:{GPU}")
torch.cuda.set_device(GPU)
print(f"torch cuda current device: {torch.cuda.current_device()}")


# Model instantiation

efficientNetModel = "EfficientNet-B3"

# TODO: put the below in model1.py instead so you don't have to write it in every script that instantiates an EfficientNet() model
if efficientNetModel == "EfficientNet-B7":
    parameters = {"num_labels": 8, "width_coefficient": 2.0, "depth_coefficient": 3.1, "dropout_rate": 0.5}
    resolution = 600

elif efficientNetModel == "EfficientNet-B6":
    parameters = {"num_labels": 8, "width_coefficient": 1.8, "depth_coefficient": 2.6, "dropout_rate": 0.5}
    resolution = 528

elif efficientNetModel == "EfficientNet-B5":
    parameters = {"num_labels": 8, "width_coefficient": 1.6, "depth_coefficient": 2.2, "dropout_rate": 0.4}
    resolution = 456

elif efficientNetModel == "EfficientNet-B4":
    parameters = {"num_labels": 8, "width_coefficient": 1.4, "depth_coefficient": 1.8, "dropout_rate": 0.4}
    resolution = 380

elif efficientNetModel == "EfficientNet-B3":
    parameters = {"num_labels": 8, "width_coefficient": 1.2, "depth_coefficient": 1.4, "dropout_rate": 0.3}
    resolution = 300

elif efficientNetModel == "EfficientNet-B2":
    parameters = {"num_labels": 8, "width_coefficient": 1.1, "depth_coefficient": 1.2, "dropout_rate": 0.3}
    resolution = 260

elif efficientNetModel == "EfficientNet-B1":
    parameters = {"num_labels": 8, "width_coefficient": 1.0, "depth_coefficient": 1.1, "dropout_rate": 0.2}
    resolution = 240

elif efficientNetModel == "EfficientNet-B0":
    parameters = {"num_labels": 8, "width_coefficient": 1.0, "depth_coefficient": 1.0, "dropout_rate": 0.2}
    resolution = 224

model = model1.EfficientNet(num_labels=parameters["num_labels"], width_coefficient=parameters["width_coefficient"], 
                            depth_coefficient=parameters["depth_coefficient"], 
                            dropout_rate=parameters["dropout_rate"]).to(device)


# Loading weights

modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220208-161939/efficientNetBestReciprocalMSE_165727167543.3294"
model.load_state_dict(torch.load(modelPath))


# Choose metrics

# NOTE: in hindsight, this probably didn't need to be done before I wrote code for test-time augmentation and full inference on testLoader but it will come in handy 
# eventually

metrics = {
    'MeanSquaredError': MeanSquaredError(),
    'MeanAbsoluteError': MeanAbsoluteError(),
}


# MUCH LATER: Test-time augmentation like in https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook
# MODERATELY LATER: Running inferencer on entirety of testLoader
# SLIGHTLY LATER: Appropriating inference to aberration magnitudes and phi_n,m angles, rather than the real and imaginary parts of the complex-number labels

# Adapting the image-plotting bit of the inference section of https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook to instead plot inferred 
# Ronchigrams and Ronchigrams from predicted labels side by side; print predicted labels alongside actual labels

    # Load the Ronchigrams to be inferred along with their actual labels. Will probably just write testLoader from training.py from scratch and write the transforms too, 
    # since what I am using for inference might not always be what is in training.py at the time. I will try to use the relevant seed, first checking if the same image 
    # is created. Might also write a new evalLoader, too.

ronchdset = RonchigramDataset("/media/rob/hdd2/james/simulations/20_01_22/Single_Aberrations.h5")

ronchdsetLength = 100008

trainFraction = 0.7
evalFraction = 0.15
testFraction = 1 - trainFraction - evalFraction

trainLength = math.ceil(ronchdsetLength * trainFraction)
evalLength = math.ceil(ronchdsetLength * evalFraction)
testLength = ronchdsetLength - trainLength - evalLength

trainSet, evalSet, testSet = random_split(dataset=ronchdset, lengths=[trainLength, evalLength, testLength], generator=torch.Generator().manual_seed(torchSeed))

mean = 0.500990092754364
std = 0.2557201385498047

testTransform = Compose([
    ToTensor(),
    Resize(resolution, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[mean], std=[std])
])

batchSize = 8
numWorkers = 2

testLoader1 = DataLoader(testSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

testLoader2 = DataLoader(testSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

# NOTE: the below is identical code to code found in inferencer.py; I ran both expecting that they would have the same output given that they same seed was used in 
# each case, along with the same initial RonchigramDataset and train:eval:test proportions and the same transform details. If the output were the same, it would mean 
# that the test loader instantiated in inferencer.py would be different from that in training.py, meaning the test loader used in inferencer.py wouldn't instead 
# overlap with the train loader used to train the model being inferenced. However, I don't think evalLoader is necessary right now.

# batch = next(iter(testLoader))
# exampleRonch = batch[0][1]
# print(exampleRonch)

    # Just plotting some Ronchigrams from un-transformed testSet

batch = next(iter(testLoader1))
print(batch)
print(batch[0].size())

comparisonBatch = torch.clone(batch[0])

# fig, ax = plt.subplots()
# ax.axis("off")
# ax.imshow(batch[0][0], cmap="gray", interpolation="nearest")
# plt.show()

# batch = None


    # Carry out evaluation of Ronchigrams to be inferred

ronchdset.transform = testTransform
batch = next(iter(testLoader2))

model.eval()

with torch.no_grad():
    x = convert_tensor(batch[0], device=device, non_blocking=True)

    yPred = model(x)    # Batch of labels
    # print(yPred)


    # Generate "predicted" Ronchigrams from predicted labels

# So that the above labels can be used to calculate a Ronchigram using calc_ronchigram from Simulations/Primary_Simulation_1.py, said function needs the following 
# arguments: imdim, simdim, C10_mag, C12_mag, C21_mag, C23_mag, C10_ang, C12_ang, C21_ang, C23_ang, I, b, t. Cnm can be calculated from the labels, and imdim, 
# simdim, and b will be made to be the same as in Simulations/Parallel_HDF5_2.py right now (1:22pm 10/02/22).
# NOTE: for I and t ranges, since they seem relatively unimpactful right now since we are only comparing Ronchigrams by eye and Poisson noise levels are fairly low, 
# I will use the same ranges but not necessarily make sure the same random values are picked as were picked to simulate the images in question.
# TODO: code in a way to get the original I and t values from the HDF5 file.

# Copying batch of Ronchigrams to create a new tensor which has space for the batch of Ronchigrams made from the predicted labels.
# NOTE: the current copy below has dimensions for the size of the Ronchigram rescaled for testing. This will not necessarily be the initial size of the Ronchigrams 
# made of the predicted labels but it is fine since that is not necessary.
ronchPredBatch = torch.clone(comparisonBatch)
numRonchs = ronchPredBatch.size()[0]

for j in range(numRonchs):
    singleYPred = yPred[j]

    abers = [0] * 8 # Just a default list I will put Cnm and phi_nm for easy unpacking into function call later (the order matches the order in calc_Ronchigram)

    # From __getitem__ in DataLoader2.py it seems that the nth and (n + 4)th elements of a label vector returned by __getitem__ correspond to the real and imaginary 
    # parts of the relevant complex number respectively--here we have zero-indexing and n is between 0 and 3 inclusive.
    for i in range(4):
        CnmReal = singleYPred[i]
        CnmImag = singleYPred[i + 4]

        Cnm_mag, Cnm_ang = cmath.polar(CnmReal + CnmImag * 1j)  # C10 magnitude/m, phi10/rad

        abers[i] = Cnm_mag
        abers[i + 4] = Cnm_ang

    imdim = 1024    # Ronchigram initial dimensions (imdim x imdim pixels)
    simdim = 100 * 10**-3   # Convergence semi-angle/rad

    min_I = 100 * 10**-12   # Minimum quoted current/A
    max_I = 1 * 10**-9      # Maximum quoted current/A
    b = 1   # i.e. assuming that all of quoted current in Ronchigram acquisition reaches the detector
    min_t = 0.1 # Minimum Ronchigram acquisition time/s
    max_t = 1   # Maximum Ronchigram acquisition time/s

    I = random.uniform(min_I, max_I)
    t = random.uniform(min_t, max_t)

    ronch = calc_Ronchigram(imdim, simdim, *abers, I, b, t)
    print(ronch.shape)

    ronch = torch.tensor(ronch)
    print(ronch.size())

    ronch = torch.unsqueeze(ronch, 2)
    print(ronch.size())

    ronchPredBatch[j] = ronch

    # if j == 0:
    #     break

# Concatenating the torch tensors whose Ronchigrams are to be compared (concatenating for easy passing to torchvision.utils.make_grid)
plottedRonchs = torch.cat((comparisonBatch, ronchPredBatch), 0)
print(plottedRonchs.size())

# TODO: just swapped dimensions of index 1 and 3 to get the tensor into a more torch-like state for plotting, but must make sure to do this earlier instead to prevent image 
# transposition
plottedRonchs = torch.transpose(plottedRonchs, 1, 3)
print(plottedRonchs.size())

plt.figure(figsize=(16, 8))
plt.axis("off")
plt.title("Comparison of tested Ronchigrams with Ronchigrams from predicted labels")
_ = plt.imshow(
    vutils.make_grid(plottedRonchs[:16], padding=2, normalize=True).cpu().numpy().transpose((1, 2, 0))
)

plt.show()



# fig, ax = plt.subplots()
# ax.axis("off")
# ax.imshow(ronch, cmap="gray", interpolation="nearest")
# plt.show()

    # Plot inferred Ronchigrams alongside predicted ones, maybe a 
    # row of the former above a row of the latter

    # Print actual labels for the above alongside predicted labels








# Print actual and predicted labels



# Inferencer after which a Ronchigram is produced from predicted labels



# MODERATELY LATER Maybe use Pandas to plot lots of inference results on mass

# MODERATLEY LATER Add a method to save things if desired
# PACKAGES

import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
import ignite # Installed via "conda install ignite -c pytorch"
import model1
import datetime

# If haven't done already, run "conda install -c conda-forge tensorboardx==1.6"

# For data loading onward
import sys
import h5py
import cmath
import math
import torchvision.transforms.functional as F2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision import utils

# For optimiser onward
from itertools import chain
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

# For update_fn definition onward
from ignite.utils import convert_tensor

# TODO: import remaining modules here as required


# Version checking

print(f"torch version: {torch.__version__}, ignite version: {ignite.__version__}")



# NAVIGATING THE TERMINAL TO THE WORKING DIRECTORY THIS FILE IS IN

os.chdir("/home/james/VSCode/cnns")
print(os.getcwd())



# SEED INFORMATION

# Arbitrary seed number
fixedSeed = 17

# Might make a way for seed to be random later
torchSeed = fixedSeed
torch.manual_seed(torchSeed)



# OPTIONS LIKE IN CNN_5.PY

# Creating this variable because in model importation I will only import EfficientNet-B7 if this name in string form is 
# what the below variable is assigned to
efficientNetModel = "EfficientNet-B0"



# GPU STUFF

GPU = 0
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.current_device()}")



# MODEL INSTANTIATION
if efficientNetModel == "EfficientNet-B7":
    model = model1.EfficientNet(num_labels=8, width_coefficient=2.0, depth_coefficient=3.1, 
                                dropout_rate=0.5).to(device)

elif efficientNetModel == "EfficientNet-B0":
    model = model1.EfficientNet(num_labels=8, width_coefficient=1.0, depth_coefficient=1.1, 
                            dropout_rate=0.2).to(device)

print(f"After model instantiation: {torch.cuda.memory_allocated(0)}")



# SAVING CURRENT ARCHITECTURE FOR EASY VIEWING AND REFERENCE

with open("/home/james/VSCode/cnns/modelLogging", "a") as f:
    f.write(f"\n\n{datetime.datetime.now()}\n\n")
    f.write(str(model))



# TRANSFORMS, DATASETS AND DATASET SPLITTING, AND DATA LOADERS

# Import dataset from dataLoader2.py

sys.path.insert(1, "/home/james/VSCode/DataLoading")
from DataLoader2 import RonchigramDataset

ronchdset = RonchigramDataset("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/Temp/Single_Aberrations.h5")

print(f"After ronchdset instantiation: {torch.cuda.memory_allocated(0)}")


# Apply transforms

if efficientNetModel == "EfficientNet-B7":
    resolution = 600 

elif efficientNetModel == "EfficientNet-B0":
    resolution = 224

# TODO: import function in DataLoader2.py that calculates mean and std for normalisation. The values below right now 
# are values from previous mean and std measurement, so should be roughly accurate, although this measurement was only 
# done over 32 Ronchigrams.
mean = 0.5008
std = 0.2562

trainTransform = Compose([
    ToTensor(),
    Resize(resolution, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[mean], std=[std])
])

testTransform = Compose([
    ToTensor(),
    Resize(resolution, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[mean], std=[std])
])

# TODO: figure out how to apply different transforms to individual split datasets rather than just applying one transform 
# to the overall dataset, although it doesn't matter so much right now since trainTransform and testTransform are the 
# same
ronchdset.transform = trainTransform


# Lengths for trainSet, evalSet and testSet

ronchdsetLength = len(ronchdset)

trainLength = math.ceil(ronchdsetLength * 0.7)
evalLength = math.ceil(ronchdsetLength * 0.15)
testLength = ronchdsetLength - trainLength - evalLength


# Split up dataset into train, eval and test

trainSet, evalSet, testSet = random_split(dataset=ronchdset, lengths=[trainLength, evalLength, testLength], generator=torch.Generator().manual_seed(torchSeed))

print(f"After ronchdset splitting: {torch.cuda.memory_allocated(0)}")


# Create data loaders via torch.utils.data.DataLoader

batchSize = 32
numWorkers = 2

trainLoader = DataLoader(trainSet, batch_size=batchSize, num_workers=numWorkers, shuffle=True, drop_last=True, 
                        pin_memory=True)


evalLoader = DataLoader(evalSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)


testLoader = DataLoader(testSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

print(f"After creating data loaders: {torch.cuda.memory_allocated(0)}")

# OPTIMISER

criterion = nn.L1Loss()

lr = 0.01

# TODO: make sure this, from the Kaggle webpage, is really applicable to your own data (I think it can be, though)
optimiser = optim.SGD([
    {
        "params": chain(model.stem.parameters(), model.blocks.parameters()),
        "lr": lr * 0.1,
    },
    {
        "params": model.head[:6].parameters(),
        "lr": lr * 0.2
    },
    {
        "params": model.head[6].parameters(),
        "lr": lr
    }],
    momentum=0.9, weight_decay=1e-3, nesterov=True)

lr_scheduler = ExponentialLR(optimiser, gamma=0.975)



# update_fn DEFINITION

# Initialise a variable that is used to check the below function only when this variable equals 1
i=0

def update_fn(engine, batch):
    # Only do checking below when i == 1
    global i
    i += 1

    model.train()

    x = convert_tensor(batch["ronchigram"], device=device, non_blocking=True)
    if i == 1:
        print(f"Size of x is: {x.size()}")

    print(f"After putting x onto the GPU: {torch.cuda.memory_allocated(0)}")
    
    y_pred = model(x)
    if i == 1: 
        print(f"Size of y_pred is: {y_pred.size()}")

    del x

    y = convert_tensor(batch["aberrations"], device=device, non_blocking=True)
    if i == 1: 
        print(f"Size of y is: {y.size()}")

    # print(y)
    # print(y_pred)


    # Compute loss
    loss = criterion(y_pred, y)
    print(loss)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    return {
        "batchloss": loss.item(),
    }




# CHECKING update_fn

batch = next(iter(trainLoader))

# Having memory issues so going to, in update_fn, put x on device, calculate y_pred on device, remove x from device, #
# then add y to device and then calculate loss
res = update_fn(engine=None, batch=batch)
print(res)

batch = None
torch.cuda.empty_cache()






# Output_transform definition

# Some tensorboard stuff

# Metrics for training

# Evaluator instantiation

# Setting up logger

# default_score_fn definition

# Early stopping

# Function to clear cuda cache between training and testing

# Training running

# Storing best model from training

# Closing the HDF5 file

ronchdset.close_file()
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
efficientNetModel = "EfficientNet-B7"



# GPU STUFF

GPU = 0
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.current_device()}")



# MODEL INSTANTIATION
model = model1.EfficientNet(num_labels=1, width_coefficient=2.0, depth_coefficient=3.1, 
                            dropout_rate=0.5).to(device)



# SAVING CURRENT ARCHITECTURE FOR EASY VIEWING AND REFERENCE

with open("/home/james/VSCode/cnns/modelLogging", "a") as f:
    f.write(f"\n\n{datetime.datetime.now()}\n\n")
    f.write(str(model))



# TRANSFORMS, DATASETS AND DATASET SPLITTING, AND DATA LOADERS

# Import dataset from dataLoader2.py

sys.path.insert(1, "/home/james/VSCode/DataLoading")
from DataLoader2 import RonchigramDataset

ronchdset = RonchigramDataset("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/Temp/Single_Aberrations.h5")


# Apply transforms

if efficientNetModel == "EfficientNet-B7":
    resolution = 600 

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

trainLength = math.ceil(ronchdsetLength * 0.70)
evalLength = math.ceil(ronchdsetLength * 0.15)
testLength = ronchdsetLength - trainLength - evalLength


# Split up dataset into train, eval and test

trainSet, evalSet, testSet = random_split(dataset=ronchdset, lengths=[trainLength, evalLength, testLength], generator=torch.Generator().manual_seed(torchSeed))


# Create data loaders via torch.utils.data.DataLoader

batchSize = 64
numWorkers = 2

trainLoader = DataLoader(trainSet, batch_size=batchSize, num_workers=numWorkers, shuffle=True, drop_last=True, 
                        pin_memory=True)

evalLoader = DataLoader(evalSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

testLoader = DataLoader(testSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)



# OPTIMISER

criterion = nn.MSELoss()

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



# update_fn definition

# Checking update_fn

# Output_transform definition

# Some tensorboard stuff

# Metrics for training

# Evaluator instantiation

# Setting up logger

# default_score_fn definition

# Early stopping

# Function to clearing cuda cache between training and testing

# Training running

# Storing best model from training

# Closing the HDF5 file

ronchdset.close_file()
# PACKAGES

import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
import ignite # Installed via "conda install ignite -c pytorch"
import model1
# If haven't done already, run "conda install -c conda-forge tensorboardx==1.6"

print(f"torch version: {torch.__version__}, ignite version: {ignite.__version__}")

# TODO: import remaining modules here as required
# TODO: import functions for transforms and data loading


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





# Saving current architecture for easy viewing and reference

# Transforms

# Datasets, including splitting

# Data loaders

# Optimiser

# Amp stuff

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
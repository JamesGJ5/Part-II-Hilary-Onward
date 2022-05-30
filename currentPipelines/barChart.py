# 1. Import modules

import os
import sys
import torch
import model1
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision.transforms.functional as F2
from ignite.utils import convert_tensor
import matplotlib.pyplot as plt
import datetime


# 2. Print time

# 3. Navigate to current working directory

# 4. Add RonchigramDataset to path

# 5. Device configuration (WANT TO USE GPU 1)

# 6. EfficientNet parameters dictionary

# 7. Model instantiation (see Line 154 of trendInferencer.py)

# 8. Load model weights

# 9. numLabels, batchSize, numWorkers

# 10. Aperture size

# 11. Put network into evaluation mode

# 2. For loop, for each file in the directory /media/rob/hdd1/james-gj/Simulations/forInference/30_05_22, that is

    # 1. Print time, file name

    # 2. Instantiate RonchigramDataset object
    
    # 3. Estimate mean and std of said object

    # 4. testTransform

    # 5. Initialise an MSE scalar for that between target c1,2 and predicted c1,2

    # 6. Initialise an MSE scalar for that between target phi1,2 and predicted phi1,2

    # 7. See Line 427 of trendInferencer.py; you basically want this with the following:

        # 1. Get batched Ronchigrams into the device chosen

        # 2. Complete this plan when you have done the beginnign etc.
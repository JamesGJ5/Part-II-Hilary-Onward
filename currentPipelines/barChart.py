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
from configparser import ConfigParser
import numpy as np


# 1. Seeding

fixedSeed = 17
torch.manual_seed(fixedSeed)


# 2. Print time

startTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(f'Starting running pipeline at {startTime}')


# 3. Navigate to current working directory

os.chdir("/home/james/VSCode/currentPipelines")
print(f"Current working directory: {os.getcwd()}")


# 4. Add RonchigramDataset to path

sys.path.insert(1, "/home/james/VSCode/DataLoading")
from DataLoader2 import RonchigramDataset, showBatch


# 5. Device configuration (WANT TO USE GPU 1)

GPU = 1
usingGPU = True

if not usingGPU:
    os.environ["CUDA_VISIBLE_DEVICES"]=""

device = torch.device(f"cuda:{GPU}" if usingGPU else "cpu")
print(f"Device being used by PyTorch (non-CUDA): {device}")

if usingGPU:
    torch.cuda.set_device(GPU if usingGPU else "cpu")
    print(f"torch cuda current device: {torch.cuda.current_device()}")


# 6. numLabels, batchSize, numWorkers

numLabels = 2
batchSize = 32
numWorkers = 8


# 7. EfficientNet parameters dictionary

parameters = {"num_labels": numLabels, "width_coefficient": 1.1, "depth_coefficient": 1.2, "dropout_rate": 0.3}
resolution = 260


# 8. Model instantiation (see Line 154 of trendInferencer.py)

model = model1.EfficientNet(num_labels=parameters["num_labels"], width_coefficient=parameters["width_coefficient"], 
                            depth_coefficient=parameters["depth_coefficient"], 
                            dropout_rate=parameters["dropout_rate"]).to(device)


# 9. Load model weights

modelPath = '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220425-104947/best_checkpoint_reciprocalLoss=32.4506.pt'
model.load_state_dict(torch.load(modelPath, map_location = torch.device(f"cuda:{GPU}" if usingGPU else "cpu"))["model"])


# 10. Aperture size

desiredSimdim = 30
actualSimdim = 70

apertureSize = 1024 / 2 * desiredSimdim / actualSimdim


# 11. Put network into evaluation mode

model.eval()


# 12. Chosen values

chosenVals = {"c10": False, "c12": True, "c21": False, "c23": False, "c30": False,
"c32": False, "c34": False, "c41": False, "c43": False, "c45": False, "c50": False, "c52": False, "c54": False, "c56": False,

"phi10": False, "phi12": True, "phi21": False, "phi23": False, "phi30": False,
"phi32": False, "phi34": False, "phi41": False, "phi43": False, "phi45": False, "phi50": False, "phi52": False, "phi54": False, "phi56": False
}


# 13. Scaling values

scalingVals = {
    "c10scaling": 1 / (100 * 10**-9), "c12scaling": 1 / (100 * 10**-9), "c21scaling": 1 / (300 * 10**-9), "c23scaling": 1 / (100 * 10**-9), 
    "c30scaling": 1 / (10.4 * 10**-6), "c32scaling": 1 / (10.4 * 10**-6), "c34scaling": 1 / (5.22 * 10**-6), "c41scaling": 1 / (0.1 * 10**-3), "c43scaling": 1 / (0.1 * 10**-3), "c45scaling": 1 / (0.1 * 10**-3),
    "c50scaling": 1 / (10 * 10**-3), "c52scaling": 1 / (10 * 10**-3), "c54scaling": 1 / (10 * 10**-3), "c56scaling": 1 / (10 * 10**-3),

    "phi10scaling": 1, "phi12scaling": 1 / (2 * np.pi / 2), "phi21scaling": 1 / (2 * np.pi / 1), "phi23scaling": 1 / (2 * np.pi / 3), 
    "phi30scaling": 1, "phi32scaling": 1 / (2 * np.pi / 2), "phi34scaling": 1 / (2 * np.pi / 4), "phi41scaling": 1 / (2 * np.pi / 1), "phi43scaling": 1 / (2 * np.pi / 3), "phi45scaling": 1 / (2 * np.pi / 5),
    "phi50scaling": 1, "phi52scaling": 1 / (2 * np.pi / 2), "phi54scaling": 1 / (2 * np.pi / 4), "phi56scaling": 1 / (2 * np.pi / 6)
} 


# 14. For loop, for each file in the directory /media/rob/hdd1/james-gj/Simulations/forInference/30_05_22, that is

    # 1. Print time, file name

    # 2. Instantiate RonchigramDataset object
    
    # 3. Estimate mean and std of said object

    # 4. testTransform

    # 5. DataLoader instantiation

    # 6. Initialise an MSE scalar for that between target c1,2 and predicted c1,2

    # 7. Initialise an MSE scalar for that between target phi1,2 and predicted phi1,2

    # 8. See Line 427 of trendInferencer.py; you basically want this with the following:

        # 1. Get batched Ronchigrams into the device chosen

        # 2. Complete this plan when you have done the beginnign etc.
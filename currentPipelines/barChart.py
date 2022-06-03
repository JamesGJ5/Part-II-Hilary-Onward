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
import math


# 1. Seeding

fixedSeed = 17
torch.manual_seed(fixedSeed)


# 2. Print time

startTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(f'Starting running bar chart pipeline at {startTime}')


# 3. Navigate to current working directory

os.chdir("/home/james/VSCode/currentPipelines")
print(f"Current working directory: {os.getcwd()}")


# 4. Add RonchigramDataset to path

sys.path.insert(1, "/home/james/VSCode/DataLoading")
from DataLoader2 import RonchigramDataset, getMeanAndStd2


# 5. Device configuration (WANT TO USE GPU 1)

GPU = 0
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

# NOTE: MSD stands for Mean Percentage Difference
# c12MSDVals contains value of predicted c1,2 for Ronchigrams in which c1,2 is in a different percentile range
c12MSDVals = np.array([])
phi12MSDVals = np.array([])

compute = False

if compute:

    with open('c12phi12MSDs.npy', 'wb') as f:

        # 14. A for loop for each file in the directory /media/rob/hdd1/james-gj/Simulations/forInference/30_05_22
        XList = [10 * (i + 1) for i in range(10)]

        for X in XList:

            # 1. Print time, file name

            testSetPath = f'/media/rob/hdd1/james-gj/Simulations/forInference/30_05_22/c12_{X-10}to{X}pct.h5'
            time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            print(f'Beginning {testSetPath} at {time}')


            # 2. Instantiate RonchigramDataset object
            
            testSet = RonchigramDataset(testSetPath, transform=None, complexLabels=False, **chosenVals, **scalingVals)


            # 3. Estimate mean and std of said object

            print(f"Resolution of each Ronchigram for which mean and standard deviation are calculated is {resolution}, which should equal the resolution used in training.")
            calculatedMean, calculatedStd = getMeanAndStd2(ronchdset=testSet, trainingResolution=resolution, diagnosticBatchSize=64, batchesTested=16, apertureSize=apertureSize)
            print(f'Estimated mean: {calculatedMean}\nEstimated std: {calculatedStd}')


            # 4. testTransform

            try:

                mean = calculatedMean
                std = calculatedStd

            except:

                mean = 0.5000
                std = 0.2551

            testTransform = Compose([
                ToTensor(),
                CenterCrop(np.sqrt(2) * apertureSize),
                Resize(resolution, F2.InterpolationMode.BICUBIC),
                Normalize(mean=[mean], std=[std])
            ])

            testSet.transform = testTransform


            # 5. DataLoader instantiation

            testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True, 
                                    drop_last=True)


            # 6. Initialise an MSD tensor for batch MSDs between target c1,2 and predicted c1,2

            c12MSD = 0


            # 7. Initialise an MSD tensor for batch MSDs between target phi1,2 and predicted phi1,2

            phi12MSD = 0


            # 8. See Line 427 of trendInferencer.py; you basically want this with the following:
            with torch.no_grad():
                for batchIdx, (batchedRonchs, batchedTargets) in enumerate(testLoader):


                    # 1. Get batched Ronchigrams into the device chosen

                    batchedRonchs = convert_tensor(batchedRonchs, device=device, non_blocking=True)
                    
                    predBatch = model(batchedRonchs).cpu()
                    # print(predBatch.size())

                    batchedTargets = batchedTargets.cpu()
                    # print(batchedTargets.size())


                    # 2. Calculate the MSD between predBatch and batchedTargets

                    SD_PerElement = (predBatch - batchedTargets) ** 2

                    c12BatchMSD = torch.mean(SD_PerElement[:, 0])
                    phi12BatchMSD = torch.mean(SD_PerElement[:, 1])


                    # 3. Contribute mean for this batch to old mean

                    c12MSD *= batchIdx
                    c12MSD += c12BatchMSD
                    c12MSD /= (batchIdx + 1)

                    phi12MSD *= batchIdx
                    phi12MSD += phi12BatchMSD
                    phi12MSD /= (batchIdx + 1)


                    # 4. Status updates:

                    if batchIdx % math.ceil(len(testSet) / batchSize / 10) == 0:

                        print(f'Batch index {batchIdx} done, with a batchSize of {batchSize}.')
                        print(f'c12MSD: {c12BatchMSD}')
                        print(f'phi12MSD: {phi12BatchMSD}')


            print(f'Final c12MSD: {c12MSD}')
            print(f'Final phi12MSD: {phi12MSD}')


            # 9. Wanna save c12MSD

            np.save(f, np.array([c12MSD]))
            np.save(f, np.array([phi12MSD]))


            # 10. For the bar chart

            c12MSDVals.append(c12MSD)
            phi12MSDVals.append(phi12MSD)


# 15. Getting the values from the file they are saved in if they aren't being computed in the current run

if not compute:

    with open('c12phi12MSDs.npy', 'rb') as f:

        for i in range(10):

            c12MSD = np.load(f)
            phi12MSD = np.load(f)

            c12MSDVals = np.append(c12MSDVals, c12MSD)
            phi12MSDVals = np.append(phi12MSDVals, phi12MSD)


# 16. Wanna make a bar chart

# Y = np.arange(len(XList))

# fig = plt.figure()

# ax = fig.add_axes([0, 0, 1, 1])

# ax.bar(Y + 0.00, c12MSDPerPercentileRange, color='b', width=0.25)
# ax.bar(Y + 0.00, phi12MSDPerPercentileRange, color='g', width=0.25)

# Here, x would be the percentage of its training maximum the magnitude c1,2 has
c12Percentiles = [
    '[0%, 10%)', '[10%, 20%)', '[20%, 30%)', '[30%, 40%)', '[40%, 50%)',
    '[50%, 60%)', '[60%, 70%)', '[70%, 80%)', '[80%, 90%)', '[90%, 100%)',
]

xPos = np.arange(len(c12Percentiles))

print(xPos.shape)
print(len(c12MSDVals))
# sys.exit()

plt.xticks(xPos, c12Percentiles)

plt.ylabel('Mean squared difference (in unit indicated) between predicted values')

plt.title('Bar chart showing success of predictions of each constant in Ronchigrams with varying percentages of c1,2 ' +\
    'compared to maximum c1,2 in training.')

plt.bar(xPos - 0.2, c12MSDVals, width=0.4, label='Predicted c1,2 (m)')
plt.bar(xPos + 0.2, phi12MSDVals, width=0.4, label='Predicted \u03A61,2 (rad)')

plt.legend()

plt.show()
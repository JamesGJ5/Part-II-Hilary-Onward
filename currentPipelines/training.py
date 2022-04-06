from configparser import ConfigParser
import torchvision
import torch
import torch.nn as nn
import os

import numpy as np
import ignite # Installed via "conda install ignite -c pytorch"
import model1
import datetime
import sys

import math
import torchvision.transforms.functional as F2
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from itertools import chain

import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from ignite.utils import convert_tensor
from ignite.engine import Engine, Events, create_supervised_evaluator

from ignite.metrics import RunningAverage, Loss
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import CustomPeriodicEvent

from ignite.handlers import global_step_from_engine
import logging
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
import matplotlib.pyplot as plt


# TODO: cite https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite (used for help in building an early version 
# of this pipeline)


# VERSION CHECKING

print(f"torch version: {torch.__version__}... ignite version: {ignite.__version__}...")



# NAVIGATING THE TERMINAL TO THE WORKING DIRECTORY THIS FILE IS IN

os.chdir("/home/james/VSCode/currentPipelines")
print(f"Current working directory: {os.getcwd()}")


# STUFF FOR IMPORTING CONFIGURATIONS

config = ConfigParser()
config.read("config1.ini")

# Section of config1.ini to import parameters from (section name is 2nd argument passed to terminal)
configSection = config[sys.argv[1]]


# SEED INFORMATION

# TODO: consider allowing seed to be random later
torchSeed = 17
torch.manual_seed(torchSeed)


# GPU STUFF

# TODO: 18th put config importation below
GPU = eval(configSection["GPU"])
# device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{GPU}")
torch.cuda.set_device(GPU)
print(f"torch cuda current device: {torch.cuda.current_device()}")


# MODEL OPTIONS

# Creating this variable because in model importation I will only import EfficientNet-B3 if this name in string form is 
# what the below variable is assigned to
efficientNetModel = "EfficientNet-B2"

numLabels = eval(configSection["numLabels"])

if efficientNetModel == "EfficientNet-B7":
    parameters = {"num_labels": numLabels, "width_coefficient": 2.0, "depth_coefficient": 3.1, "dropout_rate": 0.5}
    resolution = 600

elif efficientNetModel == "EfficientNet-B6":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.8, "depth_coefficient": 2.6, "dropout_rate": 0.5}
    resolution = 528

elif efficientNetModel == "EfficientNet-B5":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.6, "depth_coefficient": 2.2, "dropout_rate": 0.4}
    resolution = 456

elif efficientNetModel == "EfficientNet-B4":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.4, "depth_coefficient": 1.8, "dropout_rate": 0.4}
    resolution = 380

elif efficientNetModel == "EfficientNet-B3":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.2, "depth_coefficient": 1.4, "dropout_rate": 0.3}
    resolution = 300

elif efficientNetModel == "EfficientNet-B2":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.1, "depth_coefficient": 1.2, "dropout_rate": 0.3}
    resolution = 260

elif efficientNetModel == "EfficientNet-B1":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.0, "depth_coefficient": 1.1, "dropout_rate": 0.2}
    resolution = 240

elif efficientNetModel == "EfficientNet-B0":
    parameters = {"num_labels": numLabels, "width_coefficient": 1.0, "depth_coefficient": 1.0, "dropout_rate": 0.2}
    resolution = 224


# CHOOSING MODEL

model = model1.EfficientNet(num_labels=parameters["num_labels"], width_coefficient=parameters["width_coefficient"], 
                            depth_coefficient=parameters["depth_coefficient"], 
                            dropout_rate=parameters["dropout_rate"]).to(device)

print(f"Memory/bytes allocated after model instantiation: {torch.cuda.memory_allocated(GPU)}")


# LOADING WEIGHTS INTO MODEL

pretrainedWeights = eval(configSection["pretrainedWeights"])

if pretrainedWeights:

    # The below is whether the last lr computed in preTrainedWeights' run will be returned (see 
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html for more on this)
    # TODO: figure out how to actually implement getting last lr in state_dict like the above link says is doable
    useLast_lr = eval(configSection["useLast_lr"])

    # TODO: if you get an error about cuda weights being wrong or something, use the map_location parameter below as you 
    # did in inferencer.py
    model.load_state_dict(torch.load(configSection["weightsPath"]))


# TRANSFORMS, DATASETS AND DATASET SPLITTING, AND DATA LOADERS

# Import dataset from dataLoader2.py

sys.path.insert(1, "/home/james/VSCode/DataLoading")
from DataLoader2 import RonchigramDataset

simulationsPath = configSection["simulationsPath"]

c10, c12, c21, c23, c30, c32, c34, c41, c43, c45, c50, c52, c54, c56 = (eval(configSection[x]) for x in ["c10", "c12", \
                                    "c21", "c23", "c30", "c32", "c34", "c41", "c43", "c45", "c50", "c52", "c54", "c56"])

phi10, phi12, phi21, phi23, phi30, phi32, phi34, phi41, phi43, phi45, phi50, phi52, phi54, phi56 = (\
    eval(configSection[x]) for x in ["phi10", "phi12", "phi21", "phi23", "phi30", "phi32", "phi34", "phi41", "phi43", \
                                                                            "phi45", "phi50", "phi52", "phi54", "phi56"])

c10scaling, c12scaling, c21scaling, c23scaling, c30scaling, c32scaling, c34scaling, c41scaling, c43scaling, c45scaling, \
    c50scaling, c52scaling, c54scaling, c56scaling = (eval(configSection[x]) for x in ["c10scaling", "c12scaling", \
    "c21scaling", "c23scaling", "c30scaling", "c32scaling", "c34scaling", "c41scaling", "c43scaling", "c45scaling", \
                                                                "c50scaling", "c52scaling", "c54scaling", "c56scaling"])

phi10scaling, phi12scaling, phi21scaling, phi23scaling, phi30scaling, phi32scaling, phi34scaling, phi41scaling, \
    phi43scaling, phi45scaling, phi50scaling, phi52scaling, phi54scaling, phi56scaling = (eval(configSection[x]) for x \
    in ["phi10scaling", "phi12scaling", "phi21scaling", "phi23scaling", "phi30scaling", "phi32scaling", \
    "phi34scaling", "phi41scaling", "phi43scaling", "phi45scaling", "phi50scaling", "phi52scaling", "phi54scaling", \
    "phi56scaling"])

ronchdset = RonchigramDataset(hdf5filename=simulationsPath, complexLabels=False,

                                c10=c10, c12=c12, c21=c21, c23=c23, c30=c30, c32=c32, c34=c34, c41=c41, c43=c43, c45=c45,
                                c50=c50, c52=c52, c54=c54, c56=c56,

                                phi10=phi10, phi12=phi12, phi21=phi21, phi23=phi23, phi30=phi30, phi32=phi32, 
                                phi34=phi34, phi41=phi41, phi43=phi43, phi45=phi45, phi50=phi50, phi52=phi52,
                                phi54=phi54, phi56=phi56,

                                c10scaling=c10scaling, c12scaling=c12scaling, c21scaling=c21scaling, 
                                c23scaling=c23scaling, c30scaling=c30scaling, c32scaling=c32scaling, 
                                c34scaling=c34scaling, c41scaling=c41scaling, c43scaling=c43scaling, 
                                c45scaling=c45scaling, c50scaling=c50scaling, c52scaling=c52scaling,
                                c54scaling=c54scaling, c56scaling=c56scaling,
                                
                                phi10scaling=phi10scaling, phi12scaling=phi12scaling, phi21scaling=phi21scaling, 
                                phi23scaling=phi23scaling, phi30scaling=phi30scaling, phi32scaling=phi32scaling, 
                                phi34scaling=phi34scaling, phi41scaling=phi41scaling, phi43scaling=phi43scaling, 
                                phi45scaling=phi45scaling, phi50scaling=phi50scaling, phi52scaling=phi52scaling,
                                phi54scaling=phi54scaling, phi56scaling=phi56scaling)

print(f"Memory/bytes allocated after ronchdset instantiation: {torch.cuda.memory_allocated(GPU)}")

# I am storing the time in this variable scriptTime because I want the same time to be logged for both saving training 
# information and for the name of the file(s) training results in, i.e. model weights etc. Also, want this time to be 
# logged for information about the saving of mean and std calculated for the Normalize() transform.
scriptTime = datetime.datetime.now()

apertureSize = ronchdset[0][0].shape[0] / 2 # Radius of objective aperture in pixels
print(f"Aperture size is {apertureSize}")


# ESTIMATING MEAN AND STANDARD DEVIATION OF DATA

# If want to estimate the mean and std of the data (with transforms beside Normalize() applied) and pass said mean and 
#   std to the Normalize() transform
estimateMeanStd = eval(configSection["estimateMeanStd"])

# Optional estimation of mean and std of data to pass to torchvision.transforms.Normalize()
if estimateMeanStd:
    from DataLoader2 import getMeanAndStd2

    # NOTE: in a test, I found that completing the below without specificDevice == device was quicker than using the GPU, 
    # so I am doing the below without GPU support.
    print(f"Resolution of each Ronchigram for which mean and standard deviation are calculated is {resolution}, which should equal the resolution used in training.")
    calculatedMean, calculatedStd = getMeanAndStd2(ronchdset=ronchdset, trainingResolution=resolution, diagnosticBatchSize=64, batchesTested=130, apertureSize=apertureSize)
    print(calculatedMean, calculatedStd)


# Apply transforms

# TODO: import function in DataLoader2.py that calculates mean and std for normalisation. The values below right now 
# are values from previous mean and std measurement, so should be roughly accurate, although this measurement was only 
# done over 32 Ronchigrams.

try:
    mean = calculatedMean
    std = calculatedStd
except:
    # TODO: 17th change the below accordingly after running first estimateMeanStd on new simulations (16_02_22/Single_Aberrations.h5)
    mean = eval(configSection["defaultMean"])
    std = eval(configSection["defaultStd"])

trainTransform = Compose([
    ToTensor(),
    CenterCrop(np.sqrt(2) * apertureSize),
    Resize(resolution, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[mean], std=[std])
])

# TODO: if you ever change testTransform so it's not the same as trainTransform, make sure that testSet has self.transform = testTransform 
# rather than trainTransform
testTransform = Compose([
    ToTensor(),
    CenterCrop(np.sqrt(2) * apertureSize),
    Resize(resolution, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[mean], std=[std])
])

# TODO: figure out how to apply different transforms to individual split datasets rather than just applying one transform 
# to the overall dataset, although it doesn't matter so much right now since trainTransform and testTransform are the 
# same
ronchdset.transform = trainTransform

inputDtype = ronchdset[0][0].type()

# print(ronchdset[0])


# Lengths for trainSet, evalSet and testSet

ronchdsetLength = len(ronchdset)

print(f"Total number of Ronchigrams used: {ronchdsetLength}")

trainFraction = 14 / 17
evalFraction = 1 - trainFraction
testFraction = 0

trainLength = math.ceil(ronchdsetLength * trainFraction)
evalLength = math.ceil(ronchdsetLength * evalFraction)
testLength = ronchdsetLength - trainLength - evalLength

print(f"trainFraction:evalFraction:testFraction {trainFraction}:{evalFraction}:{testFraction}")


# Split up dataset into train, eval and test

# TODO: if you ever change testTransform so it's not the same as trainTransform, make sure that testSet has self.transform = testTransform 
# rather than trainTransform
trainSet, evalSet, testSet = random_split(dataset=ronchdset, lengths=[trainLength, evalLength, testLength], generator=torch.Generator().manual_seed(torchSeed))

print(f"Memory/bytes allocated after ronchdset splitting: {torch.cuda.memory_allocated(GPU)}")


# Create data loaders via torch.utils.data.DataLoader
# num_epochs is here to facilitate saving this information to file in code below, didn't want to move lots of code

trainBatchSize = eval(configSection["trainBatchSize"])
evalBatchSize = eval(configSection["evalBatchSize"])

numWorkers = eval(configSection["numWorkers"])

num_epochs = eval(configSection["num_epochs"])


# SAVING CURRENT ARCHITECTURE AND BATCH SIZE FOR EASY VIEWING AND REFERENCE

with open("/home/james/VSCode/currentPipelines/modelLogging", "a") as f:
    f.write(f"\n\n\n{scriptTime}")
    if not pretrainedWeights:
        f.write("\n\nSee config1 at the date and time this training run was done (see https://github.com/JamesGJ5/Part-II-Hilary-Onward) for weights used.")

    f.write(f"\n\nGPU: {GPU}, Torch seed: {torchSeed}, input datatype: {inputDtype}, numWorkers: {numWorkers}, train:eval:test {trainFraction}:{evalFraction}:{testFraction}")
    f.write(f"\nData loaded from {ronchdset.hdf5filename}\n\n")
    f.write(str(trainTransform))
    f.write("\n\n")
    f.write(str(testTransform))
    f.write("\n\n")
    f.write(efficientNetModel + ", " + str(parameters) + f", resolution: {resolution}" + \
            f", {ronchdsetLength} Ronchigrams" + \
            f", batch sizes: {trainBatchSize} for trainLoader & {evalBatchSize} for evalLoader" + \
            f", number of epochs: {num_epochs}\n\n")
    f.write(str(model))



trainLoader = DataLoader(trainSet, batch_size=trainBatchSize, num_workers=numWorkers, shuffle=True, drop_last=True, 
                        pin_memory=True)

# batch = next(iter(trainLoader))
# x = convert_tensor(batch["ronchigram"], device=device, non_blocking=True)
# xtype = x.type()
# print(f"trainLoader batch type is {xtype}")

# print(batch)



evalLoader = DataLoader(evalSet, batch_size=evalBatchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

# batch = next(iter(evalLoader))
# x = convert_tensor(batch["ronchigram"], device=device, non_blocking=True)
# xtype = x.type()
# print(f"evalLoader batch type is {xtype}")


# testLoader = DataLoader(testSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
#                         pin_memory=True)

# NOTE: the below is identical code to code found in inferencer.py; I ran both expecting that they would have the same output given that they same seed was used in 
# each case, along with the same initial RonchigramDataset and train:eval:test proportions and the same transform details. If the output were the same, it would mean 
# that the test loader instantiated in inferencer.py would be different from that in training.py, meaning the test loader used in inferencer.py wouldn't instead 
# overlap with the train loader used to train the model being inferenced.

# batch = next(iter(testLoader))
# exampleRonch = batch[0][1]
# print(exampleRonch)


# batch = next(iter(testLoader))
# x = convert_tensor(batch["ronchigram"], device=device, non_blocking=True)
# xtype = x.type()
# print(f"testLoader batch type is {xtype}")


print(f"Memory/bytes allocated after creating data loaders: {torch.cuda.memory_allocated(GPU)}")



# OPTIMISER

# j = 0
# y_predNaNsSum = 0
# targetNaNsSum = 0

torchCriterion = nn.MSELoss(reduction="none")

# criterionReduction = configSection["criterionReduction"]
criterion = eval(configSection["criterion"])

# if criterionReduction == "none":

#     redFun = eval(configSection["redFun"])

lr = eval(configSection["lr"])

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


# optimiser = optim.Adam([
#     {
#         "params": chain(model.stem.parameters(), model.blocks.parameters()),
#         "lr": lr * 0.1,
#     },
#     {
#         "params": model.head[:6].parameters(),
#         "lr": lr * 0.2
#     },
#     {
#         "params": model.head[6].parameters(),
#         "lr": lr
#     }],
#     weight_decay=1e-3)


# TODO: I have put this here to conveniently save the string to a logging file, must find a way to do this without 
# instantiating the string first
gamma = eval(configSection["gamma"])
lr_scheduler_string = f"ExponentialLR(optimiser, gamma={gamma})"
lr_scheduler = eval(lr_scheduler_string)


# update_fn DEFINITION

# Initialise a variable that is used to check the below function only when this variable equals 1
i=0

# This list will be used to create the y-axis of a loss curve later
batchlossVals = []

# Keeping track of batches gone through so know how many points to plot on x-axis 
# of loss curve later
batchesDone = 0

# Updates the weights while iterating over the training data
def update_fn(engine, batch):
    # Only do checking below when i == 1
    # global i
    # i += 1

    model.train()

    x = convert_tensor(batch[0], device=device, non_blocking=True)
    # if i == 1:
    #     print(f"Size of x is: {x.size()}")
    #     print(x.type())

    # print(x)

    # print(f"After putting x onto the GPU: {torch.cuda.memory_allocated(0)}")
    
    y_pred = model(x)
    # if i == 1: 
    #     print(f"Size of y_pred is: {y_pred.size()}")
        # print(y_pred.type())

    # del x

    y = convert_tensor(batch[1], device=device, non_blocking=True)
    # if i == 1: 
    #     print(f"Size of y is: {y.size()}")
        # print(y.type())

    # sys.exit()

    # print(y)
    # print(y_pred)

    # Compute loss
    loss = criterion(y_pred, y)

    # if criterionReduction == "none":
        
    #     loss = redFun(criterion, y_pred, y)

    # print(loss)
    # print(loss.size())

    # print(loss)
    # print(loss.size())

    # sys.exit()

    # print(loss)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    batchloss = loss.item()
    # print(batchloss)
    # print(batchloss.shape)

    # print(y)
    # print(y_pred)
    # print(batchloss)

    # sys.exit()

    # print(batchloss)

    global batchesDone
    batchesDone += 1

    global batchlossVals
    batchlossVals.append(batchloss)

    # if math.isnan(batchloss):
    #     print("batchloss:", batchloss)
    #     print("y:\n", y)
    #     print("y_pred:\n", y_pred)

    # if math.isinf(batchloss):
    #     print("batchloss:", batchloss)
    #     print("y:\n", y)
    #     print("y_pred:\n", y_pred)

    return {
        "batchloss": batchloss,
        # "validationLoss": 1
    }



# CHECKING update_fn

checkUpdate_fn = True

if checkUpdate_fn:
    batch = next(iter(trainLoader))

    # Having memory issues so going to, in update_fn, put x on device, calculate y_pred on device, remove x from device, #
    # then add y to device and then calculate loss
    res = update_fn(engine=None, batch=batch)
    # TODO: decomment the below when you want to test update_fn
    # print(res)

    # sys.exit()

    batch = None
    
torch.cuda.empty_cache()

# sys.exit()

# Output_transform definition

# See https://pytorch.org/ignite/quickstart.html for what is happening below
trainer = Engine(update_fn)

def output_transform(out):
    return out["batchloss"]

# Computes the running average of the batchloss, I believe
# This is mostly so batchloss can be displayed during training
# NOTE: below, the first mention of "output_transform" is one of RunningAverage's parameters, its argument is the 
# function defined above
RunningAverage(output_transform=output_transform).attach(trainer, "runningAvgBatchloss")



# SOME TENSORBOARD STUFF

exp_name = scriptTime.strftime("%Y%m%d-%H%M%S")
log_path = f"/media/rob/hdd2/james/training/fineTuneEfficientNet/{exp_name}"

tb_logger = TensorboardLogger(log_dir=log_path)

tb_logger.attach(trainer, log_handler=OutputHandler('training', ['runningAvgBatchloss', ]), event_name=Events.ITERATION_COMPLETED)
print("Experiment name: ", exp_name)

tb_logger.attach(trainer, log_handler=OutputHandler('training', output_transform=output_transform), event_name=Events.ITERATION_COMPLETED)

# tb_logger.attach(trainer, log_handler=OutputHandler('training', output_transform = lambda out: out["validationLoss"]), event_name=Events.ITERATION_COMPLETED)

# Learning rate scheduling
trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())


# Log optimiser parameters
tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimiser, "lr"), event_name=Events.EPOCH_STARTED)


# Interaction-wise progress bar
ProgressBar(bar_format="").attach(trainer, metric_names=['runningAvgBatchloss'])


# Epoch-wise progress bar with display of training losses
# TODO: figure out if it matters that below, metric_names' value doesn't contain a comma as it does above
ProgressBar(persist=True, bar_format="").attach(trainer, metric_names=['runningAvgBatchloss'], event_name=Events.EPOCH_STARTED,
closing_event_name=Events.EPOCH_COMPLETED)


# METRICS TO LOG TO TENSORBOARD

consts = (
"c10", "c12", "c21", "c23", "c30", "c32", "c34",
"c41", "c43", "c45", "c50", "c52", "c54", "c56", 

"phi10", "phi12", "phi21", "phi23", "phi30", "phi32", "phi34",
"phi41", "phi43", "phi45", "phi50", "phi52", "phi54", "phi56"
)

# Just a tuple saying True at indices whose aberration constants are included in the label and False at others
constsTrueFalse = (c10, c12, c21, c23, c30, c32, c34, c41, c43, c45, c50, c52, c54, c56,
                phi10, phi12, phi21, phi23, phi30, phi32, phi34, phi41, phi43, phi45, phi50, phi52, phi54, phi56)

constsInLabel = [const for i, const in enumerate(consts) if constsTrueFalse[i]]
print(f"constsInLabel: {constsInLabel}")

# TODO: don't find out how to get the exec() function above working--instead use a better one of the methods at 
# https://blog.finxter.com/how-to-dynamically-create-a-function-in-python/ in order to generate the below functions 
# succinctly at runtime.

def perElementTransform(idx, output):
    """Selects the element at index idx of each the predicted and target label and, when passed to a 
    torch.ignite.metrics object, computes the metric with respect to that element."""

    y_pred, y = output[0][idx], output[1][idx]

    return y_pred, y

def c10lossTransform(output):
    const = "c10"

    if const not in constsInLabel:
        idx = None
        print(f"const not in label")

    else:
        idx = constsInLabel.index(const)

    return perElementTransform(idx, output)

def c12lossTransform(output):
    const = "c12"

    if const not in constsInLabel:
        idx = None
        print(f"const not in label")

    else:
        idx = constsInLabel.index(const)

    return perElementTransform(idx, output)

def c21lossTransform(output):
    const = "c21"

    if const not in constsInLabel:
        idx = None
        print(f"const not in label")

    else:
        idx = constsInLabel.index(const)

    return perElementTransform(idx, output)

def c23lossTransform(output):
    const = "c23"

    if const not in constsInLabel:
        idx = None
        print(f"const not in label")

    else:
        idx = constsInLabel.index(const)

    return perElementTransform(idx, output)

def phi12lossTransform(output):
    const = "phi12"

    if const not in constsInLabel:
        idx = None
        print(f"const not in label")

    else:
        idx = constsInLabel.index(const)

    return perElementTransform(idx, output)

def phi21lossTransform(output):
    const = "phi21"

    if const not in constsInLabel:
        idx = None
        print(f"const not in label")

    else:
        idx = constsInLabel.index(const)

    return perElementTransform(idx, output)

def phi23lossTransform(output):
    const = "phi23"

    if const not in constsInLabel:
        idx = None
        print(f"const not in label")

    else:
        idx = constsInLabel.index(const)

    return perElementTransform(idx, output)

# TODO: 17th by creating custom metrics via method in https://pytorch.org/ignite/metrics.html, add a percentage error 
# (loss) per element metric.

metrics = {
    'OverallLoss': Loss(criterion),
    # 'c10Loss': Loss(criterion, output_transform=c10lossTransform),
    'c12Loss': Loss(criterion, output_transform=c12lossTransform),
    # 'c21Loss': Loss(criterion, output_transform=c21lossTransform),
    # 'c23Loss': Loss(criterion, output_transform=c23lossTransform),
    'phi12Loss': Loss(criterion, output_transform=phi12lossTransform),
    # 'phi21Loss': Loss(criterion, output_transform=phi21lossTransform),
    # 'phi23Loss': Loss(criterion, output_transform=phi23lossTransform)
    # 'RootMeanSquaredError': RootMeanSquaredError(),
    # 'MeanAbsoluteError': MeanAbsoluteError(),
}


# EVALUATOR INSTANTIATION

# Creating two evaluators to compute metrics on evaluation/test images and log them to Tensorboard
# Below, create_supervised_evaluator is used to facilitate the logging of the metrics, as in https://pytorch.org/ignite/quickstart.html
trainEvaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
# testEvaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)



# SETTING UP LOGGER

from ignite.contrib.handlers import CustomPeriodicEvent

# # Below, creating a custom periodic event that occurs every 3 epochs
# cpe = CustomPeriodicEvent(n_epochs=1)

# # Below, making sure that this custom periodic event is attached to the trainer Engine
# cpe.attach(trainer)

# Defining a function that permits evaluation on evalLoader (to be used now) and testLoader (not really to be used for now)
def run_evaluation(engine):
    trainEvaluator.run(evalLoader)
    # testEvaluator.run(testLoader)


# NOTE: Evaluation occurs at every 3rd epoch, I believe, starting with the first I think--this may be why there has always been a waiting 
# period before training begins. It could be that changing to EPOCHS_3_COMPLETED might be better but not so sure, it may be that the 
# second line is doing that.
trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluation)

# # Hover over Events and you see that Events.COMPLETED means that run_evaluation here is being triggered when the engine's (trainer's) run is 
# # completed, so after the final epoch. This is worth keeping, of course.
# trainer.add_event_handler(Events.COMPLETED, run_evaluation)


# Logging metrics for evaluation on evalLoader
tb_logger.attach(trainEvaluator, log_handler=OutputHandler(tag="training", metric_names=list(metrics.keys()),
global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

# Logging metrics for evaluation on TestLoader
# tb_logger.attach(testEvaluator, log_handler=OutputHandler(tag="test", metric_names=list(metrics.keys()), 
# global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

# Setup engine & logger
def setup_logger(logger):
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())



# CHECKPOINTING

# Implementing a way to show this script that the best model is the one with the lowest MeanSquaredError value
def default_score_fn(engine):

    score = 1 / engine.state.metrics['OverallLoss']

    return score


# TODO: If this script ends up creating a different number of models than 3, may need to change n_saved below from 3 to 
# something else. The below will result in a file with a number in it that corresponds to 1/MSE (so higher number means 
# better model). There may be an error with float("inf"), will wait and see if ModelCheckpoint works with it.

# TODO: rather than checkpointing the model at 3 points, maybe checkpoint at lots of points, depending on how much space there 
# is and how long this takes. Also, make sure the model present at the end of training is saved too.
best_model_handler = ModelCheckpoint(dirname=log_path, filename_prefix="best", n_saved=num_epochs, score_name="reciprocalLoss",
score_function=default_score_fn)

# Each time n_epochs (see earlier in script for what n_epochs is) epochs end, I believe checkpointing is done if the 
# score function is low enough; the dictionary below provides the model's state at that point.
trainEvaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model,})



# EARLY STOPPING

es_patience = 10
es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
# I haven't looked far into it, it doesn't seem to matter too much right now, but it may be that it is worth replacing 
# test_evaluator below with train_evaluator, if that is a better indicator of whether early stopping is worth it
trainEvaluator.add_event_handler(Events.COMPLETED, es_handler)
setup_logger(es_handler.logger)



# FUNCTION TO CLEAR CUDA CACHE BETWEEN TRAINING AND TESTING

def empty_cuda_cache(engine):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
trainEvaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
# testEvaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)



# ACTUAL TRAINING

# This is where training begins
# Note: when training, in the display for a given epoch, while the epoch is running, x/y shows the number of batches 
# iterated over, y being the total number of batches and x being the number of batches iterated over so far in the epoch.
# So, x/y shows progress of iterations in an epoch.
# TODO: see if, when the epoch ends, y changes to a number that doesn't correctly show the number of batches overal..
trainer.run(trainLoader, max_epochs=num_epochs)



# SAVING MORE TRAINING INFORMATION

# TODO: 18th below import lossCriterionString from config and replace 
# line saying "myMAFE" below with the value of lossCriterionString.
criterionName = configSection["criterion"]

with open("/home/james/VSCode/currentPipelines/modelLogging", "a") as f:
    f.write(f"\n\nTraining finished at {datetime.datetime.now()}")
    f.write("\n\n")
    f.write(str(optimiser))
    f.write("\n\nCriterion: " + str(criterion))
    f.write("\n\nLearning rate scheduler: " + lr_scheduler_string)
    f.write(f"\n\nEarly stopping patience: {es_patience}")
    try:
        f.write("\n\nTraining metrics: " + str(list(metrics.keys())))
    except:
        f.write("\n\nTraining metrics from ignite could not be logged.")
    f.write("\n\nChanges made since last training run:")
    f.write("\nChanged estimateMeanStd from False to True")
    f.write(f"\nLoss criterion is {criterionName}")
    f.write("\nMade changes mentioned on GitHub for training.py on 17/02/22 (after 11:35pm, which is when I began penultimate run)")
    f.write("\nChanged num_epochs to 4 for a short run.")



# RESULTS OF FINETUNING
# train_eval dataset metrics
print(f"trainEvaluator metrics: {trainEvaluator.state.metrics}")

# test dataset metrics
# print(f"testEvaluator metrics: {testEvaluator.state.metrics}")


# HIGHLIGHTING THE BEST-SCORE MODEL FROM TRAINING

renamingBestModel = True

if renamingBestModel:
    os.system(f"ls {log_path}")

    # NOTE: out why print(next(os.walk(log_path)))[1] returns just an empty list - this 
    # is because the ".pt" files really are just files, not directories (although their 
    # icons make them look like directories). I think they are zip files.

    # Takes all files in the directory log_path and its sub-directories
    checkpoints = next(os.walk(log_path))[2]
    print("\n" + str(checkpoints))

    # Keeps only the files ending in ".pt"
    checkpoints = sorted(filter(lambda f: f.endswith(".pt"), checkpoints))
    print("\n" + str(checkpoints))

    # TODO: automate calculation of the indices of the file names to take the 
    # scores from below

    scores = [eval(c[26:-3]) for c in checkpoints]
    print("\nScores:", scores)

    # Taking the highest reciprocal loss
    bestEpoch = np.argmax(scores)
    print("\n" + str(bestEpoch))

    if not checkpoints:
        print(f"\nThere are no weight files in {log_path}")

    else:
        # The suffix of the file name is the loss corresponding to the copied file
        modelPath = f"{log_path}/efficientNetLowestLoss_{1 / scores[bestEpoch]}"
        print("\nNew best model weights path:", modelPath)

        currentBestModelPath = os.path.join(log_path, checkpoints[bestEpoch])
        print("\n" + currentBestModelPath)

        os.system(f"cp -r {currentBestModelPath} {modelPath}")

    removeOtherFiles = False

    if removeOtherFiles:
        # Rather than remove everything in log_path, just want to remove the files 
        # that aren't the very best model weights. Don't want to remove everything 
        # in log_path because the very best model weights are stored there, so want 
        # to keep that.
        # TODO: as they arise, add other non-essential files to automatically delete, 
        # but not the loss curves, of course.
        os.system(f"rm {log_path}/best_model_test* {log_path}/events*")


with open(f"{log_path}/README.txt", "w") as f:
    f.write(f"trainEvaluator metrics: {trainEvaluator.state.metrics}")
    # f.write(f"\n\ntestEvaluator metrics: {testEvaluator.state.metrics}")


# PLOTTING AND SAVING LOSS CURVE

def lossCurve(batchAxis, batchlossAxis):
    """Plots a loss curve and saves it to log_path for a given batchAxis and batchlossAxis.
    
    batchAxis: batch axis data, currently an np.ndarray below
    batchlossAxis: batchloss axis data, currently a list below
    """

    plt.plot(batchAxis, batchlossAxis)
    plt.xlabel("Batch Number")
    plt.ylabel(f"Batch Loss ( {criterion} )")
    plt.savefig(f"{log_path}/lossCurve.png")

lossCurve(np.linspace(1, batchesDone, batchesDone).astype(int), batchlossVals)
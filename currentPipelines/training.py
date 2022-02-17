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
from modifyMAPE import modifiedMAPE

# For update_fn definition onward
from ignite.utils import convert_tensor

# For Output_transform and tensorboard stuff definition onward
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import RunningAverage, Loss, MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
from ignite.contrib.handlers import ProgressBar

# For logger onward
from ignite.contrib.handlers import CustomPeriodicEvent
from ignite.handlers import global_step_from_engine
import logging

# Model checkpointing onward
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan

# For loss curve
import matplotlib.pyplot as plt

# TODO: import remaining modules here as required

# TODO: cite the below link
# https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite was used for help in building an early version of this pipeline


# Version checking

print(f"torch version: {torch.__version__}, ignite version: {ignite.__version__}")



# NAVIGATING THE TERMINAL TO THE WORKING DIRECTORY THIS FILE IS IN

os.chdir("/home/james/VSCode/currentPipelines")
print(f"Current working directory: {os.getcwd()}")



# SEED INFORMATION

# Arbitrary seed number
fixedSeed = 17

# Might make a way for seed to be random later
torchSeed = fixedSeed
torch.manual_seed(torchSeed)



# OPTIONS LIKE IN CNN_5.PY

# Creating this variable because in model importation I will only import EfficientNet-B7 if this name in string form is 
# what the below variable is assigned to
efficientNetModel = "EfficientNet-B3"
pretrainedWeights = False
estimateMeanStd = False  # If want to estimate the mean and std of the data (with transforms beside Normalize() applied) and pass said mean and std to the Normalize() 
                        # transform



# GPU STUFF

GPU = 0
# device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{GPU}")
torch.cuda.set_device(GPU)
print(f"torch cuda current device: {torch.cuda.current_device()}")

# MODEL INSTANTIATION

numLabels = 7

# TODO: put the below in model1.py instead so you don't have to write it in every script that instantiates an EfficientNet() model
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

model = model1.EfficientNet(num_labels=parameters["num_labels"], width_coefficient=parameters["width_coefficient"], 
                            depth_coefficient=parameters["depth_coefficient"], 
                            dropout_rate=parameters["dropout_rate"]).to(device)

print(f"Memory/bytes allocated after model instantiation: {torch.cuda.memory_allocated(GPU)}")



# TRANSFORMS, DATASETS AND DATASET SPLITTING, AND DATA LOADERS

# Import dataset from dataLoader2.py

sys.path.insert(1, "/home/james/VSCode/DataLoading")
from DataLoader2 import RonchigramDataset

ronchdset = RonchigramDataset("/media/rob/hdd1/james-gj/Simulations/16_02_22/Single_Aberrations.h5", complexLabels=False)

print(f"Memory/bytes allocated after ronchdset instantiation: {torch.cuda.memory_allocated(GPU)}")

# I am storing the time in this variable scriptTime because I want the same time to be logged for both saving training 
# information and for the name of the file(s) training results in, i.e. model weights etc. Also, want this time to be 
# logged for information about the saving of mean and std calculated for the Normalize() transform.
scriptTime = datetime.datetime.now()


# Optional estimation of mean and std of data to pass to torchvision.transforms.Normalize()
if estimateMeanStd:
    from DataLoader2 import getMeanAndStd2

    # NOTE: in a test, I found that completing the below without specificDevice == device was quicker than using the GPU, 
    # so I am doing the below without GPU support.
    print(f"Resolution of each Ronchigram for which mean and standard deviation are calculated is {resolution}, which should equal the resolution used in training.")
    calculatedMean, calculatedStd = getMeanAndStd2(ronchdset=ronchdset, trainingResolution=resolution, batchesTested=320)


# Apply transforms

# TODO: import function in DataLoader2.py that calculates mean and std for normalisation. The values below right now 
# are values from previous mean and std measurement, so should be roughly accurate, although this measurement was only 
# done over 32 Ronchigrams.

try:
    mean = calculatedMean
    std = calculatedStd
except:
    # TODO: 17th change the below accordingly after running first estimateMeanStd on new simulations (16_02_22/Single_Aberrations.h5)
    mean = 0.500990092754364
    std = 0.2557201385498047

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

inputDtype = ronchdset[0][0].type()

# print(ronchdset[0])


# Lengths for trainSet, evalSet and testSet

ronchdsetLength = len(ronchdset)

print(f"Total number of Ronchigrams used: {ronchdsetLength}")

trainFraction = 0.7
evalFraction = 0.15
testFraction = 1 - trainFraction - evalFraction

trainLength = math.ceil(ronchdsetLength * trainFraction)
evalLength = math.ceil(ronchdsetLength * evalFraction)
testLength = ronchdsetLength - trainLength - evalLength

print(f"trainFraction:evalFraction:testFraction {trainFraction}:{evalFraction}:{testFraction}")


# Split up dataset into train, eval and test

trainSet, evalSet, testSet = random_split(dataset=ronchdset, lengths=[trainLength, evalLength, testLength], generator=torch.Generator().manual_seed(torchSeed))

print(f"Memory/bytes allocated after ronchdset splitting: {torch.cuda.memory_allocated(GPU)}")


# Create data loaders via torch.utils.data.DataLoader
# num_epochs is here to facilitate saving this information to file in code below, didn't want to move lots of code

batchSize = 16
numWorkers = 2

num_epochs = 11


# SAVING CURRENT ARCHITECTURE AND BATCH SIZE FOR EASY VIEWING AND REFERENCE

with open("/home/james/VSCode/currentPipelines/modelLogging", "a") as f:
    f.write(f"\n\n\n{scriptTime}")
    if not pretrainedWeights:
        f.write("\n\nSee model1.py at the date and time this training run was done (see https://github.com/JamesGJ5/Part-II-Hilary-Onward) for weights used.")

    f.write(f"\n\nGPU: {GPU}, Torch seed: {torchSeed}, input datatype: {inputDtype}, numWorkers: {numWorkers}, train:eval:test {trainFraction}:{evalFraction}:{testFraction}")
    f.write(f"\nData loaded from {ronchdset.hdf5filename}\n\n")
    f.write(str(trainTransform))
    f.write("\n\n")
    f.write(str(testTransform))
    f.write("\n\n")
    f.write(efficientNetModel + ", " + str(parameters) + f", resolution: {resolution}" + f", {ronchdsetLength} Ronchigrams"+ f", batch size: {batchSize}" + f", number of epochs: {num_epochs}\n\n")
    f.write(str(model))



trainLoader = DataLoader(trainSet, batch_size=batchSize, num_workers=numWorkers, shuffle=True, drop_last=True, 
                        pin_memory=True)

# batch = next(iter(trainLoader))
# x = convert_tensor(batch["ronchigram"], device=device, non_blocking=True)
# xtype = x.type()
# print(f"trainLoader batch type is {xtype}")

# print(batch)



evalLoader = DataLoader(evalSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

# batch = next(iter(evalLoader))
# x = convert_tensor(batch["ronchigram"], device=device, non_blocking=True)
# xtype = x.type()
# print(f"evalLoader batch type is {xtype}")


testLoader = DataLoader(testSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

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

criterion = modifiedMAPE(reduction="mean")


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

# TODO: I have put this here to conveniently save the string to a logging file, must find a way to do this without 
# instantiating the string first
lr_scheduler_string = "ExponentialLR(optimiser, gamma=0.975)"
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


    # print(loss)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    batchloss = loss.item()

    return {
        "batchloss": batchloss,
    }



# CHECKING update_fn

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

trainer = Engine(update_fn)

def output_transform(out):
    return out["batchloss"]

# Computes the running average of the batchloss, I believe
# This is mostly so batchloss can be displayed during training
# NOTE: below, the first mention of "output_transform" is one of RunningAverage's parameters, its argument is the 
# function defined above
RunningAverage(output_transform=output_transform).attach(trainer, "batchloss")



# SOME TENSORBOARD STUFF

exp_name = scriptTime.strftime("%Y%m%d-%H%M%S")
log_path = f"/media/rob/hdd2/james/training/fineTuneEfficientNet/{exp_name}"

tb_logger = TensorboardLogger(log_dir=log_path)

tb_logger.attach(trainer, log_handler=OutputHandler('training', ['batchloss', ]), event_name=Events.ITERATION_COMPLETED)
print("Experiment name: ", exp_name)

# Learning rate scheduling
trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())


# Log optimiser parameters
tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimiser, "lr"), event_name=Events.EPOCH_STARTED)


# Interaction-wise progress bar
ProgressBar(bar_format="").attach(trainer, metric_names=['batchloss',])


# Epoch-wise progress bar with display of training losses
ProgressBar(persist=True, bar_format="").attach(trainer, metric_names=['batchloss'], event_name=Events.EPOCH_STARTED,
closing_event_name=Events.EPOCH_COMPLETED)



# METRICS TO LOG TO TENSORBOARD

# TODO: 17th by creating custom metrics via method in https://pytorch.org/ignite/metrics.html,
#   add a percentage error (loss) per element metric; add a MAE per element metric;
#   add a RMSE per element metric
#   If desired, can create 
metrics = {
    'Loss': Loss(criterion),
    'RootMeanSquaredError': RootMeanSquaredError(),
    'MeanAbsoluteError': MeanAbsoluteError(),
}



# EVALUATOR INSTANTIATION

# Creating two evaluators to compute metrics on train/test images and log them to Tensorboard
trainEvaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
testEvaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)



# SETTING UP LOGGER

from ignite.contrib.handlers import CustomPeriodicEvent

cpe = CustomPeriodicEvent(n_epochs=3)
cpe.attach(trainer)

def run_evaluation(engine):
    trainEvaluator.run(evalLoader)
    testEvaluator.run(testLoader)


# Evaluation occurs after the 3rd epoch begins, I believe
trainer.add_event_handler(cpe.Events.EPOCHS_3_STARTED, run_evaluation)
trainer.add_event_handler(Events.COMPLETED, run_evaluation)


# Logging metrics for evaluation on evalLoader
tb_logger.attach(trainEvaluator, log_handler=OutputHandler(tag="training", metric_names=list(metrics.keys()),
global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

# Logging metrics for evaluation on TestLoader
tb_logger.attach(testEvaluator, log_handler=OutputHandler(tag="test", metric_names=list(metrics.keys()), 
global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

import logging

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

    score = engine.state.metrics['Loss']

    return score


# TODO: If this script ends up creating a different number of models than 3, may need to change n_saved below from 3 to 
# something else. The below will result in a file with a number in it that corresponds to 1/MSE (so higher number means 
# better model). There may be an error with float("inf"), will wait and see if ModelCheckpoint works with it.

# TODO: rather than checkpointing the model at 3 points, maybe checkpoint at lots of points, depending on how much space there 
# is and how long this takes. Also, make sure the model present at the end of training is saved too.
best_model_handler = ModelCheckpoint(dirname=log_path, filename_prefix="best", n_saved=3, score_name="Loss",
score_function=default_score_fn)
testEvaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model,})



# EARLY STOPPING

es_patience = 10
es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
# I haven't looked far into it, it doesn't seem to matter too much right now, but it may be that it is worth replacing 
# test_evaluator below with train_evaluator, if that is a better indicator of whether early stopping is worth it
testEvaluator.add_event_handler(Events.COMPLETED, es_handler)
setup_logger(es_handler.logger)



# FUNCTION TO CLEAR CUDA CACHE BETWEEN TRAINING AND TESTING

def empty_cuda_cache(engine):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
trainEvaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
testEvaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)



# ACTUAL TRAINING

# This is where training begins
# Note: when training, in the display for a given epoch, while the epoch is running, x/y shows the number of batches 
# iterated over, y being the total number of batches and x being the number of batches iterated over so far in the epoch.
# So, x/y shows progress of iterations in an epoch.
# TODO: see if, when the epoch ends, y changes to a number that doesn't correctly show the number of batches overal..
trainer.run(trainLoader, max_epochs=num_epochs)



# SAVING MORE TRAINING INFORMATION

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



# RESULTS OF FINETUNING
# train_eval dataset metrics
print(f"trainEvaluator metrics: {trainEvaluator.state.metrics}")

# test dataset metrics
print(f"testEvaluator metrics: {testEvaluator.state.metrics}")



# STORING THE BEST MODEL FROM TRAINING

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

scores = [eval(c[16:-3]) for c in checkpoints]
print("\nScores:", scores)

# TODO: check the following--I think bestEpoch is a misnomer, but for 
# some reason I have kept it called that like the Kaggle webpage I first 
# got this training pipeline from does. I think it actually just refers 
# to the best model but I could be wrong.
bestEpoch = np.argmax(scores)
print("\n" + str(bestEpoch))

if not checkpoints:
    print(f"\nThere are no weight files in {log_path}")

else:
    modelPath = f"{log_path}/efficientNetBestReciprocalMSE_{scores[bestEpoch]}"
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

# STORING A README.txt FEATURING INFORMATION ABOUT METRICS MENTIONED IN GOOGLE DOC 17/02/22
# TODO: 17th create a README and save it to same directory as weights. README must feature 
# evaluator.state.metrics if metrics was able to host all the metrics desired; if not, it 
# must host evaluator.state.metrics as well as metrics logged in other ways that couldn't 
# be logged via ignite metrics


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

    plt.show()

lossCurve(np.linspace(1, batchesDone, batchesDone).astype(int), batchlossVals)
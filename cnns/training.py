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

# For Output_transform and tensorboard stuff definition onward
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import RunningAverage, Loss, MeanAbsoluteError, MeanSquaredError
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
from ignite.contrib.handlers import ProgressBar

# For logger onward
from ignite.contrib.handlers import CustomPeriodicEvent
from ignite.handlers import global_step_from_engine
import logging

# Model checkpointing onward
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan

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

ronchdset = RonchigramDataset("/media/rob/hdd2/james/Single_Aberrations.h5")

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

print(ronchdset[0])


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

batch = next(iter(trainLoader))
# x = convert_tensor(batch["ronchigram"], device=device, non_blocking=True)
# xtype = x.type()
# print(f"trainLoader batch type is {xtype}")

print(batch)



evalLoader = DataLoader(evalSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

# batch = next(iter(evalLoader))
# x = convert_tensor(batch["ronchigram"], device=device, non_blocking=True)
# xtype = x.type()
# print(f"evalLoader batch type is {xtype}")


testLoader = DataLoader(testSet, batch_size=batchSize, num_workers=numWorkers, shuffle=False, drop_last=False, 
                        pin_memory=True)

# batch = next(iter(testLoader))
# x = convert_tensor(batch["ronchigram"], device=device, non_blocking=True)
# xtype = x.type()
# print(f"testLoader batch type is {xtype}")


print(f"After creating data loaders: {torch.cuda.memory_allocated(0)}")



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



# update_fn DEFINITION

# Initialise a variable that is used to check the below function only when this variable equals 1
i=0

def update_fn(engine, batch):
    # Only do checking below when i == 1
    global i
    i += 1

    model.train()

    x = convert_tensor(batch[0], device=device, non_blocking=True)
    if i == 1:
        print(f"Size of x is: {x.size()}")
        print(x.type())

    print(x)

    print(f"After putting x onto the GPU: {torch.cuda.memory_allocated(0)}")
    
    y_pred = model(x)
    if i == 1: 
        print(f"Size of y_pred is: {y_pred.size()}")
        # print(y_pred.type())

    del x

    y = convert_tensor(batch[1], device=device, non_blocking=True)
    if i == 1: 
        print(f"Size of y is: {y.size()}")
        # print(y.type())

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
# TODO: decomment the below when you want to test update_fn
# print(res)

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

exp_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = f"/media/rob/hdd1/james-gj/finetune_efficientnet_MNIST/{exp_name}"

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

metrics = {
    'Loss': Loss(criterion),
    'MeanSquaredError': MeanSquaredError(),
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
    MSE = engine.state.metrics['MeanSquaredError']
    # Further below, ModelCheckpoint retains the model with the highest score_function, so the score output here 
    # must be made higher for lower value of MSE, since we want to save the model with the lowest MSE
    if MSE == 0:
        score = float("inf")

    else:
        score = 1 / MSE

    return score

# TODO: If this script ends up creating a different number of models than 3, may need to change n_saved below from 3 to 
# something else. The below will result in a file with a number in it that corresponds to 1/MSE (so higher number means 
# better model). There may be an error with float("inf"), will wait and see if ModelCheckpoint works with it.
best_model_handler = ModelCheckpoint(dirname=log_path, filename_prefix="best", n_saved=3, score_name="test_recriprocal_MSE",
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

num_epochs = 20

# This is where training begins
trainer.run(trainLoader, max_epochs=num_epochs)


# RESULTS OF FINETUNING
# train_eval dataset metrics
print(f"trainEvaluator metrics: {trainEvaluator.state.metrics}")

# test dataset metrics
print(f"testEvaluator metrics: {testEvaluator.state.metrics}")



# STORING THE BEST MODEL FOR TRAINING

# Add loss curves

# Closing the HDF5 file

ronchdset.close_file()
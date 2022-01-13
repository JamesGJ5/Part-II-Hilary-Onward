import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
import random
import sys

# FIXME: IT LOOKS LIKE I HAVE WRITTEN MEANSQUAREDERROR WHERE I SHOULDN'T HAVE IN HERE, I MUST 
# AMEND THIS; I ALSO MIGHT HAVE TO TAKE OUT MENTIONS OF "ACCURACY" AS WELL, I THINK THESE APPLY TO DISCRETE LABELS.

# todo: un-comment the below when you need to install these things
# "conda activate pytorch"
# "conda install ignite -c pytorch"
# "conda install -c conda-forge tensorboardx==1.6"

import ignite

print(torch.__version__, ignite.__version__)

seed = 17
random.seed(seed)
_ = torch.manual_seed(seed)
# See https://pytorch.org/docs/stable/notes/randomness.html and
# https://www.w3schools.com/python/ref_random_seed.asp for why the above is utilised.

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class SqueezeExcitation(nn.Module):

    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(inplanes, se_planes,
                      kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(se_planes, inplanes,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


from torch.nn import functional as F


class MBConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride,
                 expand_rate=1.0, se_rate=0.25,
                 drop_connect_rate=0.2):
        super(MBConv, self).__init__()

        expand_planes = int(inplanes * expand_rate)
        se_planes = max(1, int(inplanes * se_rate))

        self.expansion_conv = None
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(inplanes, expand_planes,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
                Swish()
            )
            inplanes = expand_planes

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(inplanes, expand_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=expand_planes,
                      bias=False),
            nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
            Swish()
        )

        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)

        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_planes, planes,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )

        self.with_skip = stride == 1
        self.drop_connect_rate = torch.tensor(drop_connect_rate, requires_grad=False)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        # Add identity skip
        if x.shape == z.shape and self.with_skip:
            if self.training and self.drop_connect_rate is not None:
                self._drop_connect(x)
            x += z
        return x


from collections import OrderedDict
import math

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


class EfficientNet(nn.Module):
# See https://pytorch.org/docs/stable/generated/torch.nn.Module.html for methods the EfficientNet class inherits from 
# torch.nn.Module
    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num_channels = max(self.divisor, new_num_channels)
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

    def __init__(self, num_classes=10,
                 width_coefficient=1.0,
                 depth_coefficient=1.0,
                 se_rate=0.25,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8

        list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        list_channels = [self._setup_channels(c) for c in list_channels]

        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]

        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        # Define stem:
        self.stem = nn.Sequential(
            nn.Conv2d(1, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(list_channels[0], momentum=0.01, eps=1e-3),
            Swish()
        )

        # Define MBConv blocks
        blocks = []
        counter = 0
        num_blocks = sum(list_num_repeats)
        for idx in range(7):

            num_channels = list_channels[idx]
            next_num_channels = list_channels[idx + 1]
            num_repeats = list_num_repeats[idx]
            expand_rate = expand_rates[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            drop_rate = drop_connect_rate * counter / num_blocks

            name = "MBConv{}_{}".format(expand_rate, counter)
            blocks.append((
                name,
                MBConv(num_channels, next_num_channels,
                       kernel_size=kernel_size, stride=stride, expand_rate=expand_rate,
                       se_rate=se_rate, drop_connect_rate=drop_rate)
            ))
            counter += 1
            for i in range(1, num_repeats):
                name = "MBConv{}_{}".format(expand_rate, counter)
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append((
                    name,
                    MBConv(next_num_channels, next_num_channels,
                           kernel_size=kernel_size, stride=1, expand_rate=expand_rate,
                           se_rate=se_rate, drop_connect_rate=drop_rate)
                ))
                counter += 1

        self.blocks = nn.Sequential(OrderedDict(blocks))

        # Define head
        self.head = nn.Sequential(
            nn.Conv2d(list_channels[-2], list_channels[-1],
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(list_channels[-1], momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(list_channels[-1], num_classes)
        )

        self.apply(init_weights)

    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        y = self.head(f)
        return y

# # (width_coefficient, depth_coefficient, resolution, dropout_rate)
# 'efficientnet-b0': (1.0, 1.0, 224, 0.2),
# 'efficientnet-b1': (1.0, 1.1, 240, 0.2),
# 'efficientnet-b2': (1.1, 1.2, 260, 0.3),
# 'efficientnet-b3': (1.2, 1.4, 300, 0.3),
# 'efficientnet-b4': (1.4, 1.8, 380, 0.4),
# 'efficientnet-b5': (1.6, 2.2, 456, 0.4),
# 'efficientnet-b6': (1.8, 2.6, 528, 0.5),
# 'efficientnet-b7': (2.0, 3.1, 600, 0.5),

# if __name__ == "__main__":

# What we want this script to do
efficientet_model = "EfficientNet-B0"   # Exact model must be in this format, e.g. could also be "EfficentNet-B0"
# Note, if changing to B7, must also change the pretrained weights loaded as well as possibly something else that I have currently
# forgotten about.
will_train = True
using_existing_model_path = False
if using_existing_model_path:   # Note, this below is for cifar10
    # model_path = "efficientNet_cifar10_9475.p.pt" # Can be whatever the existing model path is, make sure cwd is My_CNNs
    model_path = "efficientNet_MNIST_9855.p.pt"
running_inferencer = False   # Be careful, setting this to False may cause issues for the bits after inferencer.run

# DEVICE CONFIGURATION
GPU = 0
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

# The below is so that I don't use a GPU while someone else is using it. I will still ensure, however, that I check 
# with others beforehand, and that I have the Terminal windows open (watch nvidia-smi and htop) like Kevin suggested.
# It seems that really torch.cuda.memory_allocated() tells you how much memory is occupied by Torch Tensors, not how
# much GPU memory is in use for all possible activities, so this is just a secondary assurance.
assert torch.cuda.memory_allocated(GPU) == 0

# Added assurance that the GPU is unused at the moment.
assert torch.cuda.memory_reserved(GPU) == 0

# MODEL IMPORTATION
# todo: remove .to(device) if it happens the code being copied doesn't need it
if efficientet_model == "EfficientNet-B0":
    model = EfficientNet(num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2).to(device)

elif efficientet_model == "EfficientNet-B7":
    model = EfficientNet(num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2).to(device)

# fout = open("EfficientNet_B0_Model_Architecture_CNN_3", "w")
# fout.write(str(model))
# fout.close()

# fout = open("EfficientNet_B0_Model_Weights_CNN_3", "w")
# fout.write(str(model.))
# fout.close()

# LOAD PRETRAINED WEIGHTS
from collections import OrderedDict



# # todo: must download the file for efficientnet_b7 weights instead
# # the file below has the pretrained weights for efficientnet-b0
# model_state = torch.load("efficientnet-b0-08094119.pth")
# # Be careful. "efficientnet..." only works if terminal's wd is /home/james/VSCode/My_CNNs. If wd was 
# # /home/james/VSCode, you'd have to put "My_CNNs/efficient..."

# # A basic remapping is required
# mapping = {
#     k:v for k, v in zip(model_state.keys(), model.state_dict().keys())
# }
# mapped_model_state = OrderedDict([
#     (mapping[k], v) for k, v in model_state.items()
# ])

# model.load_state_dict(mapped_model_state, strict=False)
# # The load_state_dict method is what copies the parameters and buffers from state_dict (here, mapped_model_state) 
# # into the model.

from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize, RandomAffine
from torchvision.transforms import ToTensor, Normalize

from torch.utils.data import Subset
import torchvision.utils as vutils

# FIXME: in their code they seem to use ! here for downloads, see if you can implement that

# N.B. instead of importing from PIL.Image import BICUBIC and using that in the transform, I used
# F2.InterpolationMode.BICUBIC to prevent a UserWarning
import torchvision.transforms.functional as F2

path = "."
if efficientet_model == "EfficientNet-B0":
    resolution = 224
elif efficientet_model == "EfficientNet-B7":
    resolution = 600

image_size = resolution
train_transform = Compose([
    Resize(image_size, F2.InterpolationMode.BICUBIC),
    RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.08,
    1.02), shear=2, fill=(124)),
    # todo: had to use fill cf. fillcolor because fillcolor deprecated, check if was OK to use same arguments
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.485], std=[0.229])
    # Note, not sure if std depends on resolution, it may do, since these were written for EfficientNet-B0, which has 
    # a resolution of 224. # todo: check
])

# # This is here 
# model.stem[0] = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False).to(device)
# model.stem[1] = nn.BatchNorm2d(1, momentum=0.01, eps=1e-3).to(device)

test_transform = Compose([
    Resize(image_size, F2.InterpolationMode.BICUBIC),
    ToTensor(),
    Normalize(mean=[0.485], std=[0.229])
])

# todo: I have set download=True rather than False for the below, different from link
# train_dataset = CIFAR10(root=path, train=True, transform=train_transform,
#                         download=True)
# test_dataset = CIFAR10(root=path, train=False, transform=test_transform,
#                         download=True)
train_dataset = MNIST(root=path, train=True, transform=train_transform,
                        download=True)
test_dataset = MNIST(root=path, train=False, transform=test_transform,
                        download=True)

import random

train_eval_indices = [random.randint(0, len(train_dataset) - 1)
for i in range(len(test_dataset))]
train_eval_dataset = Subset(train_dataset, train_eval_indices)

# todo: try applying train_eval_dataset etc. to CNN_6.py (in the My_CNNs on your laptop)

print(len(train_dataset), len(test_dataset), len(train_eval_dataset))

# todo: remember, can't do all of the above on just your laptop

from torch.utils.data import DataLoader

batch_size = 64
num_workers = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size,
num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size,
num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

# Finetuning model to CIFAR-10
# todo: "As we are interested to finetune the model to CIFAR-10, we will replace the classification fully-connected layer
#   (ImageNet-1000 vs CIFAR-10)", maybe this is something along the lines of what was wrong, but may not be. Maybe
#   could later change this fine-tuning to my simulations instead.

print((model.head[6].in_features, model.head[6].out_features))    # Yields (1280, 1000) as expected

# Don't need to change the below line for MNIST because MNIST has 10 classes (each equal to a digit between 0 and 9)
model.head[6] = nn.Linear(1280, 1).to(device)
# c10classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
                # "ship", "truck")
# MNISTclasses = (0, 1, 2, 3, 4, 5, 6, 7,
#                 8, 9)

print((model.head[6].in_features, model.head[6].out_features))  # Yields (1280, 10) as expected

assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
torch.backends.cudnn.benchmark = True   # todo: look into what this does

# Criterion and optimiser to be cross-entropy and SGD respectively
# Model parameters to be split into 2 groups: feature extractor (pretrained weights), classifier (random weights)

from itertools import chain

import torch.optim as optim
import torch.nn.functional as F

criterion = nn.MSELoss()

lr = 0.01

# todo: be careful, link spells it "optimizer"
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

from torch.optim.lr_scheduler import ExponentialLR
lr_scheduler = ExponentialLR(optimiser, gamma=0.975)

# Automatic Mixed Precision

use_amp = False

if use_amp:
    try:
        from apex import amp
        print("amp imported from apex successfully")
    except ImportError:
        print("amp NOT imported from apex successfully - see below comments")
        # Look at https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook and 
        # https://github.com/NVIDIA/apex and https://stackoverflow.com/questions/19042389/conda-installing-upgrading-directly-from-github
        # and https://stackoverflow.com/questions/46076754/use-package-from-github-in-conda-virtual-environment

    # Initialize Amp
    model, optimiser = amp.initialize(model, optimiser, opt_level="O2", num_losses=1)

# Next, let's define a single iteration function update_fn. This function is then used by ignite.engine.Engine to
# update model while running over the input data.

from ignite.utils import convert_tensor, to_onehot

i=0
def update_fn(engine, batch):
    """Used by ignite.engine.Engine to update model while running over the input data."""
    model.train()

    global i
    i += 1   # Just put this here because I only want the print statements below to work once

    x = convert_tensor(batch[0], device=device, non_blocking=True)
    if i==1: print(x.size())

    y = convert_tensor(batch[1], device=device, non_blocking=True)
    y = y.reshape((y.size(dim=0), 1))
    y = y.to(torch.float32)
    if i == 1: print(f"y's size is {y.size()}")

    # It seems I removed salient code for prediction etc. because I was simply debugging. I will now re-add relevant 
    # code here from https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook
    y_pred = model(x)

    # Compute loss (criterion is defined somewhere above this function definition)
    loss = criterion(y_pred, y)

    optimiser.zero_grad()

    # Won't be using apex amp from now on so the if statement probably won't run again
    if use_amp:
        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimiser.step()

    return {
        "batchloss": loss.item(),
    }

# Checking update_fn
batch = next(iter(train_loader))
res = update_fn(engine=None, batch=batch)

batch = None

torch.cuda.empty_cache()

print(res) # Returns 2.3156 (to 5 significant figures), which is close to the value in the webpage I am using to write this code (2.3139 to 5sf)

# Now to continue. Will have to come back and comment out the above that isn't necessary, especially since the recent above was just 
# a check that update_fn worked.

# Defining a trainer and adding some practical handlers, including a log to tensorboard (losses, metrics, lr), a 
# progress bar and models/optimizers checkpointing

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import RunningAverage, Loss, TopKCategoricalAccuracy, MeanAbsoluteError

from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler

trainer = Engine(update_fn)

def output_transform(out):
    return out["batchloss"]

RunningAverage(output_transform=output_transform).attach(trainer, "batchloss")  # Attach attaches current metric to provided enginer

# todo: learn more about ignite generally, maybe watch a YouTube video

from datetime import datetime

exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
# log_path = f"/tmp/finetune_efficientnet_cifar10/{exp_name}" (this is what it was in kaggle but I want to use HDD1)
log_path = f"/media/rob/hdd1/james-gj/finetune_efficientnet_MNIST/{exp_name}"

if will_train:
    tb_logger = TensorboardLogger(log_dir=log_path)

    tb_logger.attach(trainer, log_handler=OutputHandler('training', ['batchloss', ]), event_name=Events.ITERATION_COMPLETED)
    print("Experiment name: ", exp_name)    # Prints an experiment name equal to the correct approximate date and time

# Setting up learning rate scheduling

trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())

# Log optimiser parameters
if will_train:
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimiser, "lr"), event_name=Events.EPOCH_STARTED)
# Be careful because optimiser is assigned twice in this script, the second time for amp from apex, which you will soon
# not be using. It might be worth sorting this out, and also changing your spelling of optimiser to optimizer.

from ignite.contrib.handlers import ProgressBar

# Interaction-wise progress bar
ProgressBar(bar_format="").attach(trainer, metric_names=['batchloss',])

# Epoch-wise progress bar with display of training losses
ProgressBar(persist=True, bar_format="").attach(trainer, metric_names=['batchloss'], event_name=Events.EPOCH_STARTED,
closing_event_name=Events.EPOCH_COMPLETED)

# Creating two evaluators to compute metrics on train/test images and log them to Tensorboard

metrics = {
    'Loss': Loss(criterion),
    'MeanSquaredError': MeanAbsoluteError()
}

evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

from ignite.contrib.handlers import CustomPeriodicEvent

cpe = CustomPeriodicEvent(n_epochs=3)
cpe.attach(trainer)

# print(f"eval_train_loader's size is {eval_train_loader.size()}")
# print(f"test_loader's size is {test_loader.size()}")
# print(f"next(iter(eval_train_loader))[1]'s size is {next(iter(eval_train_loader))[1]}")
# print(f"next(iter(train_loader))[1]'s size is {next(iter(train_loader))[1]}")


def run_evaluation(engine):
    train_evaluator.run(eval_train_loader)
    evaluator.run(test_loader)

trainer.add_event_handler(cpe.Events.EPOCHS_3_STARTED, run_evaluation)
trainer.add_event_handler(Events.COMPLETED, run_evaluation)

from ignite.handlers import global_step_from_engine

# Log train eval metrics:
if will_train:
    tb_logger.attach(train_evaluator, log_handler=OutputHandler(tag="training", metric_names=list(metrics.keys()),
global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)
# Note, I have switched global_step_transform from what it was before (another_engine, which I think is part of the old 
# ignite documentation), so if this all doesn't work at the end, that may be a reason

# Log val metrics
if will_train:
    tb_logger.attach(evaluator, log_handler=OutputHandler(tag="test", metric_names=list(metrics.keys()), 
global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)
# Note, I have switched global_step_transform from what it was before (another_engine, which I think is part of the old 
# ignite documentation), so if this all doesn't work at the end, that may be a reason

# Setting up the best model checkpointing, early stopping:

import logging

# Setup engine & logger
def setup_logger(logger):
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
# es_handler._logger will later be passed to setup_logger as logger
# es_handler = EarlyStopping(...)
# EarlyStopping is imported from ignite.handlers

from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan

trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

# Store the best model
def default_score_fn(engine):
    score = engine.state.metrics['MeanSquaredError']
    return score

best_model_handler = ModelCheckpoint(dirname=log_path, filename_prefix="best", n_saved=3, score_name="test_acc",
score_function=default_score_fn)
evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model,})

# Add early stopping
es_patience = 10
es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, es_handler)
setup_logger(es_handler.logger)
# Note, the webpage I am adapting code from uses es_handler._logger. However, es_handler is an EarlyStopper object that doesn't have 
# an attribute _logger, but has an attributed logger: https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping

# No issues so far, I was worried there'd be issues with logger.addHandler in setup_logger function definition. 
# However, the module logging was imported before that function definition, and I believe the method addHandler 
# comes from that: https://docs.python.org/3/library/logging.html

# Clear cuda cache between training/testing
def empty_cuda_cache(engine):
    torch.cuda.empty_cache()
    import gc
    gc.collect()
# Note, it is probably fine that gc is imported in the function, function only needs to be called once if there 
# is just one training and one testing. However, in interesting pipelines where there are many, importing gc 
# globally would probably be much better for time complexity.

trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
# Note, the webpage this code is from passes empty_cuda_cache (a function) without parentheses, as above. I think 
# this is okay since we are only adding an event handler, and not necessarily asking it to empty the cache, but 
# if things don't work out, the lack of parentheses for empty_cuda_cache above (and maybe even for the other above 
# argument) may be the reason.
evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

num_epochs = 4

if __name__ == "__main__":

    ## Note: uncomment this when want to train
    if will_train:
        trainer.run(train_loader, max_epochs=num_epochs)
        # Final epoch has batchloss of 0.268 on the run I just did (23/11/2021, just finished circa 3:30pm). batchloss, I 
        # believe, is CrossEntropyLoss(), so this seems quite good.

        # Note: progress bar gets printed twice for the same epoch - the first time it goes to 5%, at least for Epochs 1, 2 and 3
        # Note: take heed of the user warnings
        # Note: for some of the progress bars, the x/y bit to the right starts of as x/400, but eventually becomes 20/20 - could be a formatting 
        # issue, i.e. 20/20 = 1, as does 400/400, but that doesn't mean there were only 20 images trained on with in the epoch.

        # Saving the new parameters
        # torch.save(model.state_dict(), "trained_efficientnet_b0_weights_cifar10.pth")


        # Finetuning the results

        # Test dataset
        print(evaluator.state.metrics)

        # Training subset
        print(train_evaluator.state.metrics)

    # Webpage says "Obviously, our training settings is not the optimal one and the delta between our result and the paper's one is about 5%."

    # Inference

    # Going to load the best model and recompute evaluation metrics on test dataset with a very basic Test-Time-Augmentation to 
    # boost the performances

    # Find the last checkpoint

    # Remember, if want to do this without training (i.e. things already saved to previous log_path, make sure the above 
    # uses the relevant log_path rather than a new one, given that log_path changes with time, as it has the current date 
    # and time in the title of one of its files)
    if will_train:
        os.system(f"ls {log_path}") # Unlike the webpage I am adapting code from, I use os.system() rather than !
        checkpoints = next(os.walk(log_path))[2]
        checkpoints = sorted(filter(lambda f: f.endswith(".pt"), checkpoints))
        # The Kaggle webpage has the above with .pth instead of .pt - mine save as .pth for some reason
        scores = [c[22:28] for c in checkpoints]    # c[22:28] is supposed to pick the score out of the file name
        print(scores)
        best_epoch = np.argmax(scores)
        print(best_epoch, scores)
        if not checkpoints:
            print("No weight files in {}".format(log_path))
        else:
            model_path = f"efficientNet_MNIST_{scores[best_epoch]}.pt"
            os.system(f"cp {os.path.join(log_path, checkpoints[best_epoch])} {model_path}")

        print(model_path)
        os.system(f"touch model_path_{model_path}")
        os.system(f"rm {log_path}/*")

    best_model = EfficientNet()
    best_model.load_state_dict(torch.load(model_path))

    metrics = {
        "Accuracy": Accuracy(),
        "Precision": Precision(average=True),
        "Recall": Recall(average=True)
    }

    all_pred = np.empty((0, 10), float)

    def inference_update_with_tta(engine, batch):
        global all_pred
        best_model.eval()
        with torch.no_grad():
            x, y = batch
            # Computing final prediction as a mean of predictions on x and flipped x
            y_pred1 = best_model(x)
            y_pred2 = best_model(x.flip(dims=(-1, )))
            y_pred = 0.5 * (y_pred1 + y_pred2)
            # calc softmax for submission
            curr_pred = (0.5 * (F.softmax(y_pred1, dim=-1) + F.softmax(y_pred2, dim=-1))).data.cpu().numpy()
            # Note: for the above line, the above webpage adds F.softmax(y_pred1, dim=-1) to itself rather than 
            # F.softmax(y_pred2, dim=-1). Given the comment two above this one, I think  the second y_pred1 in the 
            # the webpage's version of the above line was a typo. However, if their results are good, maybe their typo is a 
            # good fudge.
            # Note: .data doesn't seem to be recognised when typing it, might be an old method. Actually, softmax returns a 
            # Torch Tensor, which does seem to have the method .data, though it doesn't seem to do much other than return the 
            # same tensor returned by the softmax, so maybe the webpage I am adapting code from meant to do something else.
            all_pred = np.vstack([all_pred, curr_pred])

            return y_pred, y

    inferencer = Engine(inference_update_with_tta)

    for name, metric in metrics.items():
        metric.attach(inferencer, name)

    ProgressBar(desc="Inference").attach(inferencer)

    if running_inferencer:
        result_state = inferencer.run(test_loader, max_epochs=1)

        print(result_state.metrics)

    # Plot some images from training stage
    # Note: these images may be from the training stage, but were used in the evaluation section of it. Also, remember, 
    # if will_train == False and you're using a model you haven't just created in the same run, the images may differ 
    # from what would appear if it was all in the same run, although may not be the case if shuffle==True for dataset in 
    # test_loader

    # Going to test best model now to display its prowess

    batch = next(iter(test_loader))

    plt.figure(figsize=(16,8))
    plt.axis("off")
    # todo: figure out what the above is for
    plt.title("Images from Training Stage")
    # See comment under "Plot some images from training stage"
    _ = plt.imshow(
        vutils.make_grid(batch[0][:16], padding=2, normalize=True).cpu().numpy().transpose((1, 2, 0))
    )
    plt.show()

    # Classify
    best_model.eval()
    with torch.no_grad():
        y_pred = best_model(batch[0][1:2])
        # Note: be very careful, because if you change 1:2 to 1, you will get an error because y_pred will be 3D rather than 
        # 4D.

    # Print predictions
    print('-----')
    for idx in torch.topk(y_pred, k=9)[1].squeeze(0).tolist():
        prob = torch.softmax(y_pred, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=MNISTclasses[idx], p=prob*100))

    print(all_pred.shape)
    import pandas as pd
    sub = pd.DataFrame(all_pred, columns=MNISTclasses)
    sub.to_csv('efficientNetB0.csv', index_label='id')
    print(sub.head())

    # Clean up folders
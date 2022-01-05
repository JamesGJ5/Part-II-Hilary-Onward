import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
import random

# Potential issues:
#   - A UserWarning about output sizes not being the same size as targets; could be within update_fn, could be within 
#   eval_train_loader or test_loader in run_evaluation()
#   - Must cite https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook as well as the papers that 
#   code was compiled from
#   - n_saved
#   - Must implement a way to get the script to select the model with the lowest MSE
#   - STORING BEST MODEL (WEIGHTS) FROM TRAINING - here, there's a possibility that this doesn't store the weights 
#   desirably, although it should. Will have to check again for the Ronchigram simulations model
#   - The comment I have put above ModelChekckpoint
#   - Must make sure shapes are sorted throughout, especially for the inference bit

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

if __name__ == "__main__":

    # OPTIONS

    efficientet_model = "EfficientNet-B7"
    num_output_values = 8

    will_train = True   # Just states whether training will be done in this script or not

    # If True, will replace the string in existing_model_path with the model_path in question. model_path is where the 
    # best model weights are stored.
    using_existing_model_path = False
    existing_model_path = ""

    will_remove_log_path = False    # Kept False for now, will eventually make True for debugging

    running_inferencer = False  # Just states whether inferencer.run will occur

    # DEVICE CONFIGURATION
    GPU = 0
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    # I am using "watch nvidia-smi" and "htop" to monitor the GPUs, but the below is some added assurance - if anyone 
    # has any torch Tensor on the GPU I am trying to work on, the rest of the script won't run.
    assert torch.cuda.memory_allocated(GPU) == 0
    assert torch.cuda.memory_reserved(GPU) == 0

    # MODEL IMPPORTATION
    # num_classes is a bit of a misnomer, should be num_values. In any case, this is 8 values.
    # todo: rename num_classes wherever it occurs to num_values
    model = EfficientNet(num_classes=num_output_values, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2).to(device)

    print(f"model.head in_features and out_features: {(model.head[6].in_features, model.head[6].out_features)}")
    # in_features is 1280 because the number of channels submitted to the linear layer is 1280

    # SAVING CURRENT ARCHITECTURE FOR EASY VIEWING
    fout = open("EfficientNet_B7_Architecture_CNN_5", "w")
    fout.write(str(model))
    fout.close()

    # I have not yet implemented a way to save the initial weights to a file for easy viewing but they certainly work for 
    # training the model on MNIST to recognise digits between 0 and 9, as I have shown in another script of mine, 
    # CNN_3.py

    from torchvision.datasets.cifar import CIFAR100, CIFAR10
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Pad, RandomHorizontalFlip, Resize, RandomAffine
    from torchvision.transforms import ToTensor, Normalize

    from torch.utils.data import Subset, random_split
    import torchvision.utils as vutils

    # N.B. instead of importing from PIL.Image import BICUBIC and using that in the transform, I used
    # F2.InterpolationMode.BICUBIC to prevent a UserWarning
    import torchvision.transforms.functional as F2

    path = "."

    # TODO: consider adding padding
    image_size = 600    # Resolution is 600 for EfficientNet-B7 (see further above)
    train_transform = Compose([
        Resize(image_size, F2.InterpolationMode.BICUBIC),
        RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.08,
        1.02), shear=2, fill=(124)),
        # Originally got fill as a 3-tuple from https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook; 
        # had to reduce this to the first element to fit greyscale input images.
        # todo: will have to remember that I can vary the first element, especially if it isn't as applicable as it would 
        #   be for CIFAR10, which is what the webpage I adapted this code from was tuned to. Also, that webpahe is 
        #   for image size or resolution of 224 (for EfficientNet-B0).
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485], std=[0.229])
        # Originally got mean and std as 3-tuples from https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook; 
        # had to reduce these to their first elements to fit greyscale input images.
        # todo: will have to remember that I can vary the first element of each, especially if it isn't as applicable as it would 
        #   be for CIFAR10, which is what the webpage I adapted this code from was tuned to. Also, that webpahe is 
        #   for image size or resolution of 224 (for EfficientNet-B0).
    ])

    test_transform = Compose([
        Resize(image_size, F2.InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.485], std=[0.229])
        # Originally got mean and std as 3-tuples from https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook; 
        # had to reduce these to their first elements to fit greyscale input images.
        # todo: will have to remember that I can vary the first element of each, especially if it isn't as applicable as it would 
        #   be for CIFAR10, which is what the webpage I adapted this code from was tuned to. Also, that webpahe is 
        #   for image size or resolution of 224 (for EfficientNet-B0).
    ])

    # DATA LOADING (NOTE, THIS WILL ONLY BE HERE UNTIL I MAKE MY RONCHIGRAMS INTO A DATASET THAT I LOAD USING 
    # torchvision.datasets.DatasetFolder())
    train_dataset = MNIST(root=path, train=True, transform=train_transform, download=True)
    test_dataset = MNIST(root=path, train=False, transform=test_transform, download=True)

    print(f"train_dataset and test_dataset lengths before splitting: {len(train_dataset)}, {len(test_dataset)}")

    # https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook makes the val dataset data that continues 
    # to remain in the train dataset, which doesn't make sense to me, so I implemented the below:
    train_dataset, train_eval_dataset = random_split(train_dataset, [len(train_dataset)-len(test_dataset), 
    len(test_dataset)], generator=torch.Generator().manual_seed(seed))

    print(f"train_dataset, train_eval_dataset and test_dataset lengths after splitting: {len(train_dataset)}, {len(test_dataset)}, {len(train_eval_dataset)}")

    from torch.utils.data import DataLoader

    batch_size = 125
    num_workers = 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
    num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)

    eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size,
    num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
    num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

    # REALLY SORRY, I USED APEX.AMP FOR THE TRAINING, WHICH MAY BE A NIGHTMARE TO INSTALL
    assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
    torch.backends.cudnn.benchmark = True

    from itertools import chain

    import torch.optim as optim
    import torch.nn.functional as F

    criterion = nn.MSELoss()

    lr = 0.01

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

    # NOTE, HAVEN'T ACTUALLY TRIED THINGS WITH use_amp = False
    use_amp = True

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

    # Defining a single iteration function update_fn. This function is then used by ignite.engine.Engine to
    # update model while running over the input data.

    from ignite.utils import convert_tensor, to_onehot

    i=0
    def update_fn(engine, batch):
        model.train()

        global i
        i += 1   # Just put this here because I only want the print statements below to work once

        x = convert_tensor(batch[0], device=device, non_blocking=True)
        if i == 1:
            print(x.size())

        y = convert_tensor(batch[1], device=device, non_blocking=True)
        y = y.reshape((y.size(dim=0), 1))
        y = y.to(torch.float32)
        if i == 1: 
            print(f"y's size is {y.size()}")
            print(y)

        y_pred = model(x)
        if i == 1: 
            print(f"y_pred's size is {y_pred.size()}")
            print(y_pred)

        # Compute loss
        loss = criterion(y_pred, y)
        optimiser.zero_grad()

        if use_amp:
            with amp.scale_loss(loss, optimiser, loss_id=0) as scaled_loss:
                scaled_loss.backward()

        else:
            loss.backward()

        optimiser.step()

        return {
            "batchloss": loss.item(),
        }

    # Checking update_fn
    batch = next(iter(train_loader))
    batch_batchloss = update_fn(engine=None, batch=batch)

    batch = None
    torch.cuda.empty_cache()

    print(batch_batchloss)

    # Defining a trainer and adding some practical handlers, including a log to tensorboard (losses, metrics, lr), a 
    # progress bar and models/optimizers checkpointing.

    from ignite.engine import Engine, Events, create_supervised_evaluator
    from ignite.metrics import RunningAverage, Loss, TopKCategoricalAccuracy, MeanAbsoluteError

    from ignite.contrib.handlers import TensorboardLogger
    from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler

    trainer = Engine(update_fn)

    def output_transform(out):
        return out["batchloss"]

    RunningAverage(output_transform=output_transform).attach(trainer, "batchloss")  # Attach attaches current metric to provided enginer

    from datetime import datetime

    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Going to change MNIST below for something more applicable to this project
    log_path = f"/media/rob/hdd1/james-gj/finetune_efficientnet_MNIST/{exp_name}"

    if will_train:
        # Have to make sure this is under will_train otherwise a new storage directory will be made (even though there 
        # won't be any post-training weights to go in there because there won't be any training) every time I even want 
        # to just run an existing set of weights for inference. As a result, I will put anything that uses the below later 
        # under an "if will_train" statement as well.
        tb_logger = TensorboardLogger(log_dir=log_path)

        tb_logger.attach(trainer, log_handler=OutputHandler('training', ['batchloss', ]), event_name=Events.ITERATION_COMPLETED)
        print("Experiment name: ", exp_name)    # Prints an experiment name equal to the correct approximate date and time

    # Setting up learning rate scheduling
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())

    # Log optimiser parameters
    if will_train:
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimiser, "lr"), event_name=Events.EPOCH_STARTED)

    from ignite.contrib.handlers import ProgressBar

    # Interaction-wise progress bar
    ProgressBar(bar_format="").attach(trainer, metric_names=['batchloss',])

    # Epoch-wise progress bar with display of training losses
    ProgressBar(persist=True, bar_format="").attach(trainer, metric_names=['batchloss'], event_name=Events.EPOCH_STARTED,
    closing_event_name=Events.EPOCH_COMPLETED)

    # Creating two evaluators to compute metrics on train/test images and log them to Tensorboard

    metrics = {
        'Loss': Loss(criterion),
        'MeanSquaredError': MeanAbsoluteError() # Seemed most applicable to the labels
    }

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    from ignite.contrib.handlers import CustomPeriodicEvent

    cpe = CustomPeriodicEvent(n_epochs=3)
    cpe.attach(trainer)

    def run_evaluation(engine):
        train_evaluator.run(eval_train_loader)
        test_evaluator.run(test_loader)

    trainer.add_event_handler(cpe.Events.EPOCHS_3_STARTED, run_evaluation)
    trainer.add_event_handler(Events.COMPLETED, run_evaluation)

    from ignite.handlers import global_step_from_engine

    # Log train eval metrics:
    if will_train:
        tb_logger.attach(train_evaluator, log_handler=OutputHandler(tag="training", metric_names=list(metrics.keys()),
    global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

    # Log val metrics
    if will_train:
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="test", metric_names=list(metrics.keys()), 
    global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

    # Setting up checkpointing of the best model, early stopping
    # https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/
    import logging

    # Setup engine & logger
    def setup_logger(logger):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # BEGINNING A WAY TO STORE THE BEST MODEL
    # This used to have engine.state.metrics['Accuracy'], but Accuracy() from ignite.metrics was only applicable to 
    # binary, multiclass and multilabel data, so I chose MeansSquaredErorr() instead. Implementing a way to show 
    # this script that the best model is the one with the lowest MeanSquaredError value
    def default_score_fn(engine):
        MSE = engine.state.metrics['MeanSquaredError']
        # Further below, ModelCheckpoint retains the model with the highest score_function, so the score outputted here 
        # must be made as higher for lower MSE
        if MSE == 0:
            score = float("inf")

        else:
            score = 1 / MSE

        return score

    # If this script ends up creating a different number of models than 3, may need to change n_saved below from 3 to 
    # something else. The below will result in a file with a number in it that corresponds to 1/MSE (so higher number means 
    # better model). There may be an error with float("inf"), will wait and see if ModelCheckpoint works with it.
    best_model_handler = ModelCheckpoint(dirname=log_path, filename_prefix="best", n_saved=3, score_name="test_recriprocal_MSE",
    score_function=default_score_fn)
    test_evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model,})

    # ADDING EARLY STOPPING
    es_patience = 10
    es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
    # I haven't looked far into it, it doesn't seem to matter too much right now, but it may be that it is worth replacing 
    # test_evaluator below with train_evaluator, if that is a better indicator of whether early stopping is worth it
    test_evaluator.add_event_handler(Events.COMPLETED, es_handler)
    setup_logger(es_handler.logger)

    # CLEARING CUDA CACHE BETWEEN TRAINING AND TESTING
    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    test_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
    train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    num_epochs = 20

    # TRAINING
    if will_train:
        trainer.run(train_loader, max_epochs=num_epochs)
        # Note: progress bar gets printed twice for the same epoch - the first time it goes to 5%, at least for Epochs 1, 2 and 3
        # Note: take heed of the user warnings
        # Note: for some of the progress bars, the x/y bit to the right starts of as x/400, but eventually becomes 20/20 - could be a formatting 
        # issue, i.e. 20/20 = 1, as does 400/400, but that doesn't mean there were only 20 images trained on with in the epoch.

        # RESULTS OF FINETUNING
        # train_eval dataset metrics
        print(f"train_evaluator metrics: {train_evaluator.state.metrics}")

        # test dataset metrics
        print(f"test_evaluator metrics: {test_evaluator.state.metrics}")

        # STORING BEST MODEL (WEIGHTS) FROM TRAINING
        os.system(f"ls {log_path}")
        checkpoints = next(os.walk(log_path))[2]
        checkpoints = sorted(filter(lambda f: f.endswith(".pt"), checkpoints))

        # c[x:y] is meant to return, for the files storing the checkpointed models from ModelCheckpoint, the part of the 
        #   file name with the score (equal to 1/MSE). Will find x and y by running the training and seeing what files are 
        #   made.
        scores = [c[x:y] for c in checkpoints]
        print(f"The scores, 1/MSE, are: {scores}")

        # Chooses the epoch whose number equal the index position of the highest score. Not exactly sure why they called it 
        # best_epoch in https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook, I believe it is just the 
        # best model.
        best_epoch = np.argmax(scores)
        print(f"best epoch/model: {best_epoch}, best epoch/model's score: {scores[best_epoch]}")

        if not checkpoints:
            print(f"There are no weight files in {log_path}")

        else:
            model_path = f"efficientNet_MNIST_{scores[best_epoch]}.pt"
            os.system(f"cp {os.path.join(log_path, checkpoints[best_epoch])} {model_path}")

        print(f"Best model is now being stored in model_path, i.e.: {model_path}")

        if will_remove_log_path:
            os.system(f"rm {log_path}/*")

    if using_existing_model_path:
        model_path = existing_model_path

    # INFERENCE TESTS ETC.
    best_model = EfficientNet()
    best_model.load_state_dict(torch.load(model_path))

    metrics = {
        'Loss': Loss(criterion),
        'MeanSquaredError': MeanAbsoluteError()
    }

    all_pred = np.empty((0, 8), float)

    def inference_update_with_tta(engine, batch):
        global all_pred
        best_model.eval()
        with torch.no_grad():
            x, y = batch
            # Carrying out inferences on flipped images
            y_pred = best_model(x.flip(dims=(-1, )))    # todo: need to sort out some shapes earlier on to make this work
            all_pred = np.vstack([all_pred, curr_pred])

            return y_pred, y

    inferencer = Engine(inference_update_with_tta)

    for name, metric in metrics.items():
        metric.attach(inferencer, name)

    ProgressBar(desc="Inference").attach(inferencer)

    # ACTUAL INFERENCE RUNNING BIT
    if running_inferencer:
        result_state = inferencer.run(test_loader, max_epochs=1)
        
        print(result_state.metrics)
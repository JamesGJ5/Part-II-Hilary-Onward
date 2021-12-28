import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
import random

# todo: un-comment the below when you need to install these things
# "conda activate pytorch"
# "conda install ignite -c pytorch"
# "conda install -c conda-forge tensorboardx==1.6"

import ignite

print(torch.__version__, ignite.__version__)

seed = 17
random.seed(seed)
_ = torch.manual_seed(seed)


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
            nn.Conv2d(3, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
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

    # DEVICE CONFIGURATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL IMPORTATION
    # todo: remove .to(device) if it happens the code being copied doesn't need it
    model = EfficientNet(num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2).to(device)

    def print_num_params(model, display_all_modules=False):
        total_num_params = 0
        for n, p in model.named_parameters():
            num_params = 1
            for s in p.shape:
                num_params *= s
            if display_all_modules: print("{}: {}".format(n, num_params))
            total_num_params += num_params
        print("-" * 50)
        print("Total number of parameters: {:.2e}".format(total_num_params))

    # Gives 5.29e+06, just as in https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook
    # print_num_params(model)

    # LOAD PRETRAINED WEIGHTS
    from collections import OrderedDict

    # todo: must download the file for efficientnet_b7 weights instead
    # the file below has the pretrained weights for efficientnet-b0
    model_state = torch.load("My_CNNs/efficientnet-b0-08094119.pth")

    # A basic remapping is required
    mapping = {
        k:v for k, v in zip(model_state.keys(), model.state_dict().keys())
    }
    mapped_model_state = OrderedDict([
        (mapping[k], v) for k, v in model_state.items()
    ])

    model.load_state_dict(mapped_model_state, strict=False)

    # CHECKING MODEL ON A SINGLE IMAGE

    import json

    with open("My_CNNs/labels_map.txt", "r") as h:
        labels = json.load(h)

    from PIL import Image
    import torchvision.transforms as transforms

    img = Image.open("My_CNNs/giant_panda.jpg")
    # Preprocessing image
    tfms = transforms.Compose([transforms.Resize(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456,
                                0.406], [0.229, 0.224, 0.225]),])
    # Had to put to(device) here
    x = tfms(img).unsqueeze(0).to(device)
    plt.imshow(img)
    plt.show()

    # Classify. Note: after with statement ends, y_pred remains in GPU, so have to decomment for further operations
    # model.eval()
    # with torch.no_grad():
    #     y_pred = model(x)

    # Print predictions
    print('-----')
    # for idx in torch.topk(y_pred, k=5)[1].squeeze(0).tolist():
    #     prob = torch.softmax(y_pred, dim=1)[0, idx].item()
    #     print('{label:<75} ({p:.2f}%)'.format(label=labels[str(idx)], p=prob*100))

    from torchvision.datasets.cifar import CIFAR100, CIFAR10
    from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize, RandomAffine
    from torchvision.transforms import ToTensor, Normalize

    from torch.utils.data import Subset
    import torchvision.utils as vutils

    # FIXME: in their code they seem to use ! here for downloads, see if you can implement that

    # N.B. instead of importing from PIL.Image import BICUBIC and using that in the transform, I used
    # F2.InterpolationMode.BICUBIC to prevent a UserWarning
    import torchvision.transforms.functional as F2

    path = "."
    image_size = 224
    train_transform = Compose([
        Resize(image_size, F2.InterpolationMode.BICUBIC),
        RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.08,
        1.02), shear=2, fill=(124, 117, 104)),
        # todo: had to use fill cf. fillcolor because fillcolor deprecated, check if was OK to use same arguments
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = Compose([
        Resize(image_size, F2.InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # todo: I have set download=True rather than False for the below, different from link
    train_dataset = CIFAR10(root=path, train=True, transform=train_transform,
                            download=True)
    test_dataset = CIFAR10(root=path, train=False, transform=test_transform,
                           download=True)

    import random

    train_eval_indices = [random.randint(0, len(train_dataset) - 1)
    for i in range(len(test_dataset))]
    train_eval_dataset = Subset(train_dataset, train_eval_indices)

    # todo: try applying train_eval_dataset etc. to CNN_6.py (in the My_CNNs on your laptop)

    print(len(train_dataset), len(test_dataset), len(train_eval_dataset))

    # todo: remember, can't do all of the above on just your laptop

    from torch.utils.data import DataLoader

    batch_size = 125
    num_workers = 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
    num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
    num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

    eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size,
    num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

    # Plot some training images
    batch = next(iter(train_loader))

    plt.figure(figsize=(16, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        vutils.make_grid(batch[0][:16], padding=2, normalize=True).numpy().transpose((1, 2, 0))
    )
    # FIXME: it seems the images shown are not all of the same size, figure out why, probably to do with the transform
    plt.show()

    # Classify prior to fine tuning. Note, y_pred gets stored on GPU because model is on GPU
    # model.eval()
    # with torch.no_grad():
    #     y_pred = model(batch[0][:1].to(device)) # I think batch[0][:1] selects the first image

    # Print predictions
    print('-----')
    # for idx in torch.topk(y_pred, k=9)[1].squeeze(0).tolist():
    #     prob = torch.softmax(y_pred, dim=1)[0, idx].item()
    #     print('{label:<75} ({p:.2f}%)'.format(label=labels[str(idx)], p=prob * 100))

    # batch = None
    torch.cuda.empty_cache()

    # Finetuning model to CIFAR-10
    # todo: "As we are interested to finetune the model to CIFAR-10, we will replace the classification fully-connected layer
    #   (ImageNet-1000 vs CIFAR-10)", maybe this is something along the lines of what was wrong, but may not be. Maybe
    #   could later change this fine-tuning to my simulations instead.

    print((model.head[6].in_features, model.head[6].out_features))    # Yields (1280, 1000) as expected

    model.head[6] = nn.Linear(1280, 10)
    c10classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
                  "ship", "truck")

    print((model.head[6].in_features, model.head[6].out_features))  # Yields (1280, 10) as expected

    assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
    torch.backends.cudnn.benchmark = True   # todo: look into what this does

    # Criterion and optimiser to be cross-entropy and SGD respectively
    # Model parameters to be split into 2 groups: feature extractor (pretrained weights), classifier (random weights)

    from itertools import chain

    import torch.optim as optim
    import torch.nn.functional as F

    criterion = nn.CrossEntropyLoss
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

    use_amp = False
    # todo: see https://github.com/NVIDIA/apex for information

    # FIXME: sort out amp below, see above link https://github.com/NVIDIA/apex. I think it only works on earlier Python
    #   version and/or LINUX - you are going to be using LINUX so may be easy to implement. Once the below works, set
    #   use_amp above to True
    # if use_amp:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         os.system("conda activate pytorch1")
    #         os.system("git clone https://github.com/NVIDIA/apex")
    #         os.system("conda install -v --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext'")
    #         from apex import amp
    #
    # # Initialize Amp
    # model, optimiser = amp.initialize(model, optimiser, opt_level="O2", num_losses=1)

    # Next, let's define a single iteration function update_fn. This function is then used by ignite.engine.Engine to
    # update model while running over the input data.

    from ignite.utils import convert_tensor

    def update_fn(engine, batch):
        model.train()
        print("1")
        x = convert_tensor(batch[0], device=device, non_blocking=True)
        y_pred = model(x)

        print("2")
        y = convert_tensor(batch[1], device=device, non_blocking=True)
        print("3")
        # Compute loss
        loss = criterion(y_pred, y)
        print("4")
        optimiser.zero_grad()
        if use_amp:
            with amp.scale_loss(loss, optimiser, loss_id=0) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        print("5")
        optimiser.step()
        print("6")
        return {
            "batchloss": loss.item(),
        }

    print("Here now")
    print()
    # Checking update_fn
    # batch = next(iter(train_loader))
    print("And now we are here")
    res = update_fn(engine=None, batch=batch)

    # FIXME: my GPUs don't have the RAM to perform the above, I must do this on the group computer instead
    print("And here")
    batch = None
    torch.cuda.empty_cache()

    print(res)









    # todo: figure out what above is arbitrary and can be changed to improve accuracy etc. Implement features from
    #   CS231n.


import h5py
import math
import random
import cmath
import matplotlib.pyplot as plt
# Called F2 because in CNN_6.py there's another importation named F, want to be consistent
import torchvision.transforms.functional as F2
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision import utils

# "Let’s create a dataset class for our face landmarks dataset. We will read the csv in __init__ but leave the reading of images to __getitem__. This is memory efficient because all the images are not stored in the memory at once but read as required."
# TODO: think of similar ways to maximise efficiency

# TODO: make sure to implement a way to easily check if things are working as in the tutorial and use it in incremental development.

# TODO: think about what transforms must be implemented in the data loading stage, i.e. which aren't sufficiently implemented
# in the training/testing pipeline already

# TODO: make sure to test not only the dataset itself but torch.utils.data.DataLoader too, perhaps in a similar way to 
# how the pytorch.org tutorial displays the batched data near the end

# TODO: assert whatever must be asserted, e.g. number of labels equals number of images

# TODO: currently the Ronchigrams have dimensions height x width; currently, I believe, the train/test pipe line takes 
# only Ronchigrams of 3 dimensions, with the additional dimension being the number of colour channels, equal to 1. 
# I won't code this into the RonchigramDataset class definition, since this is dependent on train/test pipeline, which 
# is subject to change; instead I will leave this to the transforms themselves, probably.

# Seed 17 is arbitrary here
seed = 17
random.seed(seed)

# Plan:

# Making sure that the data can be read correctly at all from the HDF5 file
TestLoading = False

if TestLoading:
    with h5py.File("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/Temp/Single_Aberrations.h5", "r") as f:
        RandMags = f["random_mags dataset"]
        RandAngs = f["random_angs dataset"]
        RandI = f["random_I dataset"]
        Randt = f["random_t dataset"]
        ronchs = f["ronch dataset"]

        # Might want to test other datasets in file so change the below accordingly
        dset = RandMags

        # print("\n", dset)

        # # Rank is the number assigned by MPI to the parallel process whose output was saved in each case
        # for rank in range(dset.shape[0]):
        #     print(f"\nRank {rank}:")
        #     for j in [0, math.ceil(dset.shape[1]/4), math.ceil(dset.shape[1]/2), -1]:
        #         print(dset[rank, j])

        print(ronchs[0, 5555])
        print(RandMags[0, 5555])
        print(RandAngs[0, 5555], "\n")


# Second, create the RonchigramDataset class
# - DONE Make sure the class definition is consistent with torch.utils.DataLoader by following 2:35 of 
# https://www.youtube.com/watch?v=zN49HdDxHi8:
# -- DONE Make sure __init__ downloads, reads data etc.
# -- DONE Make sure __getitem__ provides the "item" when an index of a single number is passed as an argument
# -- DONE Make sure __len__ provides the size of the data, i.e. number of images I think
# - DONE Incorporate a file open/close or with method correctly
# -- DONE I am going to open the HDF5 file when the class object is instantiated, and add a close method to the class definition
# - DONE Implement a way to make the aberration information complex
# -- DONE Going to do this under getitem

# - Incorporate a transforms option to the RonchigramDataset class definition if needs be
# -- The transforms option in the pytorch.org tutorial, in the class definition for the dataset, works upon the sample, 
#   i.e. the dictionary containing both the data and the label. However, they custom-defined their transform classes 
#   such that the transforms could select the data from said sample dictionary. I, however, will not be custom-defining 
#   my classes; instead, I will apply the transform to the Ronchigram in __getitem__ before storing the Ronchigram in 
#   the sample dictionary.
# - TODO: may eventually want both ToTensor and Normalize to happen by default in RonchigramDataset, so perhaps at some 
#   point code these in.
# - After coding any of the above, check it works

class RonchigramDataset(Dataset):
    """Ronchigram dataset loaded from a single HDF5 file (see contents of "if TestLoading" above for file contents). Labels are 
    initially aberration magnitudes and angles but a magnitude/angle pair get changed into a single complex number, the 
    magnitude/m being its modulus and the angle/rad being its argument. Currently, aberrations are C10, C12, C21 and 
    C23 in Krivanek notation."""

    def __init__(self, hdf5file: str, transform=None):
        """Args:
                hdf5file: path to the HDF5 file containing the data as mentioned in the comment under this class' definition
                transform (callable, optional): transforms being incroporated
        """
        self.f = h5py.File(hdf5file, "r")

        self.RandMags = self.f["random_mags dataset"]
        self.RandAngs = self.f["random_angs dataset"]
        self.ronchs = self.f["ronch dataset"]

        # The below is just here as a reminder that the HDF5 file also stores random Ronchigram capture current and 
        # capture time information. Decomment when necessary.
        # self.RandI = f["random_I dataset"]
        # self.Randt = f["random_t dataset"]

        self.transform = transform

    def __len__(self):
        # Note: this relies on all of the HDF5 file memory spaces being filled with valid data, e.g. no incomplete 
        # simulations
        numRanks = self.ronchs.shape[0]
        ronchsPerRank = self.ronchs.shape[1]

        numRonchs = numRanks * ronchsPerRank

        return numRonchs

    def __getitem__(self, idx):
        """idx is the single-number index referring to the item being got. Since, for each of self.RandMags, 
        self.RandAngs and self.ronchs, the first dimension is rank and only the second dimension is the item itself, 
        this method must take idx and calculate from it a corresponding rank and index within that rank itself. Here, 
        an aberration's magnitude and angle are converted to a single complex number."""

        numRanks = self.ronchs.shape[0]
        itemsPerRank = self.ronchs.shape[1]

        rank = idx // itemsPerRank
        itemInRank = idx % itemsPerRank

        ronch = self.ronchs[rank, itemInRank]

        mags = self.RandMags[rank, itemInRank]
        angs = self.RandAngs[rank, itemInRank]

        # Putting the aberrations into complex form
        # TODO: see if there is a function that does what you are doing below (i.e. that can take an array of moduli 
        # and an array of arguments and return an array of complex numbers) without you having to use a for loop

        # List of aberrations in complex form, where each element will be for C10, C12, C21, and C23 respectively, each 
        # element being a complex number whose modulus is aberration magnitude/m and whose argument is aberration phi_n,m/rad
        complexList = []

        for aber in range(len(mags)):
            complexAber = cmath.rect(mags[aber], angs[aber])

            complexList.append(complexAber)

        # The below is so that ToTensor() normalises the Ronchigram to between 0 and 1 inclusive
        ronch = ronch.astype(np.uint8)

        # Certain torchvision.transform transforms, like ToTensor(), require numpy arrays to have 3 dimensions 
        # (H x W x C) rather than 2D (as of 5:01pm 08/01/22), hence the below
        if ronch.ndim == 2:
            ronch = np.expand_dims(ronch, 2)

        if self.transform:
            ronch = self.transform(ronch)

        # Decomment if you go back to using the magnitudes and angles themselves as labels
        # sample = {"ronchigram": ronch, "aberration magnitudes": mags, "aberration angles": angs}

        sample = {"ronchigram": ronch, "aberrations": complexList}

        return sample

    def close_file(self):
        self.f.close()

ronchdset = RonchigramDataset("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/Temp/Single_Aberrations.h5")

# print(ronchdset.RandMags)
# print(ronchdset.RandAngs)
# print(ronchdset.ronchs)
# print(len(ronchdset))
# print(ronchdset[5555])
# print(ronchdset.f)
# The below is only here because I wanted to check if the close_file() method could be called twice without error, 
# and it indeed can
# ronchdeset.close_file()
# Just wanted to see whether the expand_dims method I had applied to the Ronchigram in the RonchigramDataset definition 
# was working properly, it seems it is.
# print(ronchdset[0]["ronchigram"].shape)
# print(ronchdset[0]["ronchigram"])
# print(np.amax(ronchdset[0]["ronchigram"]).dtype)
# ToTensor() requires numpy array to have datatype uint8 to normalise to values between 0 and 1, so I use the below to 
# check the datatype
# print(ronchdset[0]["ronchigram"].dtype)

# ronchdset.close_file()


# DONE Third, check that a bunch of samples can be plotted properly (without transforms)

# Defining a function that allows images to be plotted; this will probably be useful not only in testing RonchigramDataset 
# before implement torch.utils.data.DataLoader but also afterwards too

def show_data(ronch, abers):
    """Show a Ronchigram along with the values of the aberrations it contains."""

    # So, when I first tried this function, matplotlib plotted it in colour despite the Ronchigrams having dimensions 
    # 1024 x 1024. Matplotlib applies a default colourmap when there are only 2 dimensions, this doesn't necessarily 
    # mean the Ronchigram itself is of colour. Later, transforms will add a colour channel dimension, whose element = 1, 
    # so need not worry too much. For now, will just plot a greyscale colourmap like in Simulations/Primary_Simulation_1.py.
    # TODO: in later simulations, add a dimensions for the colour channel, its element equal to 1.
    # Despite having conerted the 2D array into a H x W x C array in the RonchigramDataset class definition, passing the 
    # argument "gray" to the cmap parameter below is probably still necessary, especially before the Ronchigram is 
    # normalised to array elements in between 0 and 1.
    plt.imshow(ronch, cmap="gray")

    plt.xlabel(f"{abers}")

    # Slight pause so plots have time to update
    plt.pause(0.001)

# NOTE: the below might look funny because the datatype of the numpy array is changed to np.uint8 in __getitem__ so that 
# I could get ToTensor() to normalise the Ronchigrams to in between 0 and 1 inclusive
# plt.figure()
# show_data(ronchdset[0]["ronchigram"], ronchdset[0]["aberrations"])
# plt.show()

# fig = plt.figure()

# for i in range(len(ronchdset)):
#     sample = ronchdset[i]

#     print(i, sample["ronchigram"].shape, len(sample["aberrations"]))

#     ax = plt.subplot(2, 2, i + 1)
#     plt.tight_layout()
#     ax.set_title("Sample at index #{}".format(i))
#     show_data(ronchdset[i]["ronchigram"], ronchdset[i]["aberrations"])

#     if i == 3:
#         plt.show()
#         break


# Fourth, apply transforms and repeat the third step
# - I think I should indeed implement a way in the RonchigramDataset class definition to incorporate transforms
# - Once that is done, will instantiate one RonchigramDatset object, whose transforms are Resize and Normalize like 
#   in My_CNNs/CNN_6.py; I won't put ToTensor() in yet because, as far as I know, Matplotlib doesn't work with torch 
#   Tensors, and I want to check diagramatically if the transforms are working properly. Not much point in putting 
#   RandomHorizontal flip in because shouldn't do anything for images of spherical aberration Ronchigrams and for 
#   the others, angles are randomly chosen between a range that includes the angles giving rise tot he result of 
#   horizontal flips.
# - After the above, I will check that ToTensor() also works, by seeing if I can print the sizes of resulting arrays 
#   by using the .size() (NOT .size) torch Tensor method.
# - Actually, it may be the case that since I am getting the transform functions from torch, the Ronchigrams must be 
#   tensors. I will first try without ToTensor() but it may be that my first transform has to be a tensor in each case.
# - In any case, I will have to introduce the train and test transforms so I will do that below.
# - Some transforms, like ToTensor(), rely on the original numpy array having dimensions of H x W x C, as in 
#   https://pytorch.org/vision/stable/transforms.html; I will probably have to unsqueeze the numpy array in the 
#   RonchigramDataset class definition, although withou torch.unsqueeze() of course, unless it turns out that works with 
#   numpy arrays. In fact, it seems can use numpy.expand_dims as in 
# https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
# - It seems that ToTensor(), as well as converting a numpy array into a torch Tensor, results in a torch Tensor with 
#   values in between 0 and 1, so don't worry about the idea that elements with values outside of the range 0 to 1 might 
#   be interpreted as coloured. Also, it seems Normalize() doesn't limit elements to this range like I though it did.
# - OK, I plan on implementing a way to work out the mean and std of my data (see Google Drive > 4th Year > Resources > 
#   Machine Learning > Normalization for more), but the method I am looking at requires torch's DataLoader. So, I will 
#   test the transforms with a normalization mean of 0.5 and std of 0.3, fairly arbitrarily. Then, after I reach the 
#   DataLoader stage somewhere below, I will implement a way to calculate mean and std.
# - It seems ToTensor() converted my numpy arrays to torch Tensors but didn't rescale values to in between 0 and 1, 
#   probably because many original elements were outside of the range [0, 255] specified in 
#   https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor; going to see how I should 
#   rescale on my own. In fact, going to rescale numpy array to between 0 and 255 in the RonchigramDataset class 
#   definition, in __getitem__, just by dividing my the maximum element as in 
#   https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor

# Image size must be 600 x 600 for EfficientNet-B7
resolution = 600

trainTransform = Compose([
    ToTensor(),
    Resize(resolution, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[0.5], std=[0.3])
])

testTransform = Compose([
    ToTensor(),
    Resize(resolution, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[0.5], std=[0.3])
])

ronchdset.transform = trainTransform
# print(ronchdset[0]["ronchigram"])

# Using the show_landmarks_batch() function in https://pytorch.org/tutorials/beginner/data_loading_tutorial.html 
# as inspiration.

# plt.figure()

# images_batch = [ronchdset[i]["ronchigram"] for i in range(4)]
# batch_size = len(images_batch)
# im_size = images_batch[0].size(2)
# grid_border_size = 2

# grid = utils.make_grid(images_batch)
# plt.imshow(grid.numpy().transpose((1, 2, 0)))
# plt.title("Batch from dataloader")

# plt.ioff()
# plt.show()


# Fifth, check that torch.utils.data.DataLoader works on the above by adapting the third step, 
# train and test transforms incorporated

dataloader = DataLoader(ronchdset, batch_size=4,
                        shuffle=True, num_workers=0)

def showBatch(batchedSample):
    """Show Ronchigram for a batch of samples."""

    images_batch = [ronchdset[i]["ronchigram"] for i in range(4)]
    batch_size = len(images_batch)
    im_size = images_batch[0].size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title("Batch from dataloader")

# Right now I am in the middle of showing that the dataloader works, as inspired by the pytorch.org tutorial. The 
# function body above I copied from the last section, planning to adapt it. Still have to find mean and std of data 
# for Normalize, and practise splitting the datasets.




ronchdset.close_file()
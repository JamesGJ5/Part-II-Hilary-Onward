import h5py
import matplotlib.pyplot as plt
import random
import torch
import math

from torch.utils.data import Dataset

seed = 17
random.seed(seed)
_ = torch.manual_seed(seed)

class RonchigramsDataset(Dataset):
    """Dataset of Ronchigrams, where the seed = 17
random.seed(seed)
_ = torch.manual_seed(seed)data are Ronchigram image arrays 
    and the labels are 4-tuples of the magnitudes/m of the aberrations they 
    feature (C10, C12, C21 and C23, as of 12/12/2021) and 4-tuples of the 
    phi_n,m angles/rad associated with each aberration.
    
    NOTE: the Ronchigram HDF5 links file should be open in read mode for this to work. 
    Will probably change that later, however.
    """

    def __init__(self, links_file, transform=None):
        # TODO: when you have time and have sorted things out, change from links_file to 
        # better files.
        # TODO: __init__ will open links_file, so remember to close it at the end of each method definition. However, 
        # there may be an issue here in that closing it before calling another method will prevent the subsequent method 
        # from operating. For that reason, will define a method that closes the file.
        # Will try to read as much as possible from the HDF5 files from possible in __init__ without consuming too much memory. 
        # If not possible, will just leave as much reading as possible to __getitem__.
        """
        Args:
            links_file (string): Path to the HDF5 file #
                (/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/links.h5) 
                which has links to the HDF5 files from the same directory that each contain 
                Ronchigrams, magnitudes and phi_n,m angles.
            transform (callable, optional): Optional transform to be applied 
                on a sample. NOTE: should either be train_transform or test_transform.
        """
        self.transform = transform

        # Opening links_file
        self.h5fr = h5py.File(links_file, 'r')

        # Instantiating lists that will "contain" the nominal dataset in each HDF5 file that links_file has a link for; 
        # for example, the 0th element of ronch_dset_list will be a ronch_dset from one link in h5fr, the 1th element will 
        # be a ronch_dset from another, etc.
        # TODO: add I's and t's when necessary
        self.ronch_dset_list = []
        self.random_mags_dset_list = []
        self.random_angs_dset_list = []

        # Appending these datasets to the above lists, so all can be easily accessed in later methods
        for link in self.h5fr:
            f = self.h5fr[link]

            self.ronch_dset_list.append(f["ronch dataset"])
            self.random_mags_dset_list.append(f["random_mags dataset"])
            self.random_angs_dset_list.append(f["random_angs dataset"])

    def __len__(self):
        # len(dataset) is to return the size of the dataset
        # Number of Ronchigrams shall be returned as length of dataset
        ronch_number = 0
        for dset in self.ronch_dset_list:
            ronch_number += len(dset)

        return ronch_number

    def __getitem__(self, idx):
        """
        Args:
            idx: absolute index of simulation in the datasets
            # NOTE: one index value alone is required for torch.utils.data.DataLoader to work
        """
        idx1 = math.floor(idx / len(self))
        idx2 = idx % 5556   # 5556 is just the number of simulations in each linked file
        # Using idx as a single number may not be so intuitive for selecting a random thing to look at, but it will be #
        # very useful for torch.utils.data.DataLoader
        # NOTE: there are dictionaries involved here, so be wary that samples may not always be in the same 
        # position.

        ronch = self.ronch_dset_list[idx1][idx2]
        mags = self.random_mags_dset_list[idx1][idx2]
        angs = self.random_angs_dset_list[idx1][idx2]

        # sample = {"ronch": ronch, "mags": mags, "angs": angs}

        if self.transform:
            ronch = torch.squeeze(self.transform(ronch))

        labels = {"mags": mags, "angs": angs}

        return ronch, labels
        # Before all the above were entries in one dictionary, as in 
        # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        # However, since there were two labels in question, I changed this according to 
        # https://stackoverflow.com/questions/66446881/how-to-use-a-pytorch-dataloader-for-a-dataset-with-multiple-labels

    def close_file(self):
        # Closes links_file
        self.h5fr.close()

from torchvision.transforms import Compose, Pad, RandomHorizontalFlip, Resize, RandomAffine
import torchvision.transforms.functional as F2
from torchvision.transforms import ToTensor, Normalize

# NOTE: the below is here to check that it will affect my Ronchigram simulations desirably
# TODO: consider adding padding
image_size = 600
train_transform = Compose([
    ToTensor(),
    Resize(image_size, F2.InterpolationMode.BICUBIC),
    RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.08,
    1.02), shear=2, fill=(124)),
    # TODO: NOT THAT THIS IS AN OLD DATA LOADING FILE, RANDOM AFFINE SHOULD NOT BE USED!
    # Originally got fill as a 3-tuple from https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook; 
    # had to reduce this to the first element to fit greyscale input images.
    # todo: will have to remember that I can vary the first element, especially if it isn't as applicable as it would 
    #   be for CIFAR10, which is what the webpage I adapted this code from was tuned to. Also, that webpahe is 
    #   for image size or resolution of 224 (for EfficientNet-B0).
    RandomHorizontalFlip(),
    Normalize(mean=[0.485], std=[0.229])
    # Originally got mean and std as 3-tuples from https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook; 
    # had to reduce these to their first elements to fit greyscale input images.
    # todo: will have to remember that I can vary the first element of each, especially if it isn't as applicable as it would 
    #   be for CIFAR10, which is what the webpage I adapted this code from was tuned to. Also, that webpahe is 
    #   for image size or resolution of 224 (for EfficientNet-B0).
])

test_transform = Compose([
    ToTensor(),
    Resize(image_size, F2.InterpolationMode.BICUBIC),
    Normalize(mean=[0.485], std=[0.229])
    # Originally got mean and std as 3-tuples from https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook; 
    # had to reduce these to their first elements to fit greyscale input images.
    # todo: will have to remember that I can vary the first element of each, especially if it isn't as applicable as it would 
    #   be for CIFAR10, which is what the webpage I adapted this code from was tuned to. Also, that webpahe is 
    #   for image size or resolution of 224 (for EfficientNet-B0).
])

# practice = RonchigramsDataset("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/links.h5")

# fig = plt.figure()

# # I could generalise the below, but to save time, since I know there should be 18 elements in each dset list and 
# # 5556 within each dset within each dset list, I will just use these numbers in iterations below.
# to_be_transformed = True
# for idx1 in range(18):
#     # Only want example printed for idx1=0 and idx2 values of 0, 1, 2 and 3
#     if idx1 == 1:
#         break

#     for idx2 in range(5556):
#         sample = practice[idx1, idx2]
#         if to_be_transformed:
#             sample['ronch'] = torch.squeeze(train_transform(sample['ronch']))
#             # NOTE: as it is now (12/12/2021, 6:49pm, train_transform results in, for each simulation, a transformed 
#             # Ronchigram of size (1, 600, 600) - it seems plt.imshow below does not support this, so I use squeeze to 
#             # remove the dimension indicated by "1")

#         print(idx1, idx2, sample['ronch'].shape, sample['mags'].shape, sample['angs'].shape)

#         ax = plt.subplot(1, 4, idx2 + 1)
#         plt.tight_layout()
#         ax.set_title('Sample #{}{}'.format(idx1, idx2))
#         ax.axis("off")
#         plt.imshow(sample["ronch"], cmap="gray", interpolation="nearest")

#         # Only want example printed for idx2 values of 0, 1, 2 and 3
#         if idx2 == 3:
#             plt.show()
#             break

# transformed_dataset = RonchigramsDataset(
#     links_file="/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/links.h5",
#     transform=train_transform
# )

# plot_images = False
# # NOTE: if demonstration (depiction) below is going to work, only 4 data may be added to the plot below (since 
# # plt.subplot has arguments beginning with (1, 4) and thus creates a 1 by 4 grid for images to be placed upon)
# if plot_images: fig = plt.figure()

# # NOTE: below I am being lazy in just using the numbers I know apply here 
# # (there are 18 files containing simulations in question and 5556 
# # simulations in each of said files)
# for idx1 in range(18):
#     if idx1 == 1:   # Right now only want to show the example for idx1 of 0 and idx2 values of 0, 1, 2 and 3
#         break
#     for idx2 in range(5556):
#         sample = transformed_dataset[idx1, idx2]

#         print(idx1, idx2, sample['ronch'].shape, sample['mags'].shape, sample['angs'].shape)

#         if plot_images:
#             ax = plt.subplot(1, 4, idx2 + 1)
#             plt.tight_layout()
#             ax.set_title('Sample #{}{}'.format(idx1, idx2))
#             ax.axis("off")
#             plt.imshow(sample["ronch"], cmap="gray", interpolation="nearest")

#         if idx2 == 3: # Right now only want to show the example for idx1 of 0 and idx2 values of 0, 1, 2 and 3
#             if plot_images: plt.show()
#             break

# transformed_dataset.close_file()

# Now I am going to see if I can apply torch.utils.data.DataLoader. Going to look up 
# this class a little bit more to see if I can glean whether it is applicable to 
# my dataset as is or how I can adapt my dataset to make it applicable. I would 
# rather adapt my dataset than adapt torch.utils.data.DataLoader, since doing the 
# former should be much easier.

# Still doing the above. I was a bit concerned that mention of train=True and train=False in example dataloading 
# would mean my code would not be applicable to torch.utils.data.DataLoader, however it seems these are arguments 
# passed to simply downloading example datasets like MNIST, so that you can download a train set and test set from 
# separate data, I assume the format of the data is not different however. After all, DataLoader has no parameter 
# called train.

# 3:05 of https://www.youtube.com/watch?v=zN49HdDxHi8 seems to reiterate that the structure of the class 
# RonchigramsDataset must have a __getitem__(self, index) and a __len__(self). It seems probable that the lenght 
# is used for batching, and from that, indices will be given to pick Ronchigrams using __getitem__. I doubt index 
# can simply be replaced by a tuple without modifying the DataLoader source code so I will think of a work-around that 
# can allow items from all 18 files to be extracted using a single index.
# TODO: use mpi4py and parallel HDF5 to get all simulations into one file, so that one index can be used easily.

# I can probably modify my class definition of RonchigramsDataset, specifically __getitem__, to take one number as an 
# index, namely a number between 0 and len(dataset)-1. Then, idx1 will be worked out by performing math.floor(number/5556), 
# since there are 5556 simulations per linked file. idx2 can be worked out by performing number%5556. For example, if 
# number is 5555, math.floor(number/5556) == 0, which is what we want since 5555 should correspond to the last simulation in 
# the 0th linked file; 5555%5556 == 5555, which is what we want if we are seeking said simulation.

# Below is where I shall test after the above change.

tobe_traineval_set = RonchigramsDataset(
    links_file="/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/links.h5",
    transform=train_transform
)

tobe_test_set = RonchigramsDataset(
    links_file="/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/links.h5",
    transform=test_transform
)

from torch.utils.data import Subset, random_split

indices = [*range(len(tobe_traineval_set))]
print(indices[0])
random.shuffle(indices)
print(indices[0])
traineval_indices = indices[:85007]
test_indices = indices[85007:]
# Above, I first made a list featuring all the indices present in tobe_traineval_set (or tobe_test_set), then shuffled them, 
# then took slices of the shuffled list to get the indices traineval_set would occupy and that test_set would occupy. 
# Next I will take a subset of each tobe_{}_set corresponding to these indices, then use random_split to subdivide 
# the resultant traineval_set into a train_set and an eval_set. Right now, 85007 results in an approximately 85:15 ratio

traineval_set = Subset(tobe_traineval_set, traineval_indices)
print(f"traineval_set length is {len(traineval_set)}")

train_set, eval_set = random_split(traineval_set, [len(traineval_set) - 15001, 15001], generator=torch.Generator().manual_seed(seed))
print(f"train_set length is {len(train_set)}, eval_set length is {len(eval_set)}")

test_set = Subset(tobe_test_set, test_indices)
print(f"test_set length is {len(test_set)}")

# train_set = wholedataset
# train_set = Subset(wholedataset, [*range(4)])
# train_set, test_set = random_split(wholedataset, [len(wholedataset) - 2, 2], generator=torch.Generator().manual_seed(seed))
# train_set.transform = train_transform

plot_images = True
# NOTE: if demonstration (depiction) below is going to work, only 4 data may be added to the plot below (since 
# plt.subplot has arguments beginning with (1, 4) and thus creates a 1 by 4 grid for images to be placed upon)
if plot_images: fig = plt.figure()

dataset = train_set
for idx in range(len(dataset)):
    sample = dataset[idx]
    ronch = sample[0]
    mags = sample[1]["mags"]
    angs = sample[1]["angs"]

    print(idx, ronch.shape, mags.shape, angs.shape)

    if plot_images:
        ax = plt.subplot(1, 4, idx + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(idx))
        # ax.axis("off")
        plt.imshow(ronch, cmap="gray", interpolation="nearest")

    if idx == 3: # Right now only want to show the example for idx values of 0, 1, 2 and 3
        if plot_images: plt.show()
        break

# It seems switching from idx1, idx2 to a single argument (idx) passed to __getitem__ (besides self), has worked. 
# Now will try out torch.utils.data.DataLoader.
# TODO: In CNN_5.py, we have splitting into train, eval and test sets. Must make sure I can do this easily with my own 
# dataset as well, but will work on trying out the DataLoader first.

from torch.utils.data import DataLoader

# NOTE: I have configured __getitem__ such that it picks out the right sample, however, the mags & angs bit of the sample 
# may not be consistent with the format that DataLoader desires for classes, so I must be careful about this. The simplest 
# solution would probably be to extend the list of angs to the list of mags, then there would only be one value (one list) 
# for each "class".

# I have not managed to find a source that explicitly details what the format of the return value for __getitem__ in my 
# RonchigramDataset class definition must be in order to make it consistent with torch.utils.data.DataLoader, however, 
# from https://stackoverflow.com/questions/66446881/how-to-use-a-pytorch-dataloader-for-a-dataset-with-multiple-labels and 
# https://www.youtube.com/watch?v=zN49HdDxHi8&t=191s I have gleaned that the best solution would be for said method to 
# return a 2-tuple, the first element of it being the Ronchigram and the second element being a dictionary of two entries, 
# the keys being the names of the class and the values being their respective values.
# NOTE: I will hope for now that the values can be vectors, I am sure they can be.

# I have done the above and it still works well with the most recently written image-plotting example (as of 10:35am on 
# 13/12/2021). Now I will try out torch.utils.data.DataLoader as well as checking if I can split my dataset into train, 
# eval and test easily.

# The first of the above I will do will be splitting things into train, eval and test datasets. Fairly arbitrarily, I will 
# go for an 70:15:15 ratio. For the test dataset, it wouldn't make a lot of sense to apply transforms, since effects of 
# said transforms wouldn't necessarily be present in experimental Ronchigrams. However, I should probably still apply 
# some test transform as I do in CNN_5.py, e.g. to make the training data sizes consistent with the model (can do this 
# with experimental data such that model performs as well as possible). I have now loaded in test_transform and it seems to 
# work. The best course of action re: designating images for each set seems to be to instantiate a RonchigramDataset 
# object called dataset and then do the splitting, then do resultant_transform.transform = train_transform or test_transform 
# as required. It appears to not be so easy to split using slices, so I will try torch.utils.data.Subset
# It seems Subset may not be tailored to my data. I will think of a better method.
# Actually, it seems Subset does work to an extent, except for the second argument to Subset I wasn't passing the sequence 
# of indices I wanted correctly. However, it seems that applying the transform to the result of Subset doesn't work so 
# well (resizing doesn't seem to happen).

# Actually, torch.utils.data.random_split may be what I really want as opposed to Subset, I will try it.
# random_split does seem pretty good, but again it doesn't apply the transform I want and it also raises an 
# AttributeError when i try to close the file.
# TODO: for now I will instantiate two RonchigramDataset objects, one with train_transform applied and another with 
# test_transform applied. Then, I will do a get a Subset of each such that one has the training data and the other the test 
# data. Then, I will do a random_split of the former to split it into train and eval data. However, I must come back and 
# fix this fudge.

# I think the splitting and the subsets work for now, although I will of course have to go back and clean up the process. 
# Now I will move on to trying out torch.utils.data.DataLoader

batch_size = 4
num_workers = 1
train_loader = DataLoader(train_set, batch_size=batch_size,
num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)

eval_train_loader = DataLoader(eval_set, batch_size=batch_size,
num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

test_loader = DataLoader(test_set, batch_size=batch_size,
num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, type(sample_batched), len(sample_batched), len(sample_batched[0]), len(sample_batched[1]), 
    sample_batched[0], sample_batched[1])
    # len(sample_batched) is 2 because what the Data Loader does is return features and labels.
    # len(sample_batched[0]) is 4 because there are 4 Ronchigrams per batch
    # len(sample_batched[1]) is 2 because here there are two dictionary entries, the keys being "mags" and "angs" and 
    # the values being the labels for each (of which there should be 4 in each key, one for each Ronchigram)
    break

# TODO: make sure thingsa are being closed correctly
tobe_traineval_set.close_file()
tobe_test_set.close_file()


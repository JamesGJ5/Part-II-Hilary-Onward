import h5py
import math
import random
import cmath
import matplotlib.pyplot as plt
# Called F2 because in CNN_6.py there's another importation named F, want to be consistent
import torchvision.transforms.functional as F2
import numpy as np
import torch
import datetime

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision import utils
from ignite.utils import convert_tensor

import sys

# TODO: assert whatever must be asserted, e.g. number of labels equals number of images

# Creating the RonchigramDataset class
# - Incorporate a transforms option to the RonchigramDataset class definition if needs be
# - TODO: may eventually want both ToTensor and Normalize to happen by default in RonchigramDataset, so perhaps at some 
#   point code these in.
# - I am changing the class definition below in accordance with airsplay's comments (25 Jun 2020, 6 Jul 2020 and 15 Jul 
#   2020) at https://github.com/pytorch/pytorch/issues/11929, in order to avoid opening the HDF5 and forking, which will 
#   prevent reading from the file by various workers in the training pipeline from being done successfully. Among other 
#   things, modifying the dataset class includes adding an open_hdf5 method and adding a way in __init__ to calculate 
#   length for __len__
class RonchigramDataset(Dataset):
    """Ronchigram dataset loaded from a single HDF5 file (see contents of "if TestLoading" above for file contents). Labels are 
    initially aberration magnitudes and angles but a magnitude/angle pair get changed into a single complex number, the 
    magnitude/m being its modulus and the angle/rad being its argument. Currently, aberrations are C10, C12, C21 and 
    C23 in Krivanek notation."""

    def __init__(self, hdf5filename: str, transform=None, complexLabels=True, removePhi10=True):
        """Args:
                hdf5filename: path to the HDF5 file containing the data as mentioned in the comment under this class' definition
                transform (callable, optional): transforms being incroporated
                # NOTE: complexLabels should be made False, see Google doc 16/02/22
                complexLabels: whether the labels will be in complex form or not (True or False)
                removePhi10: whether the phi10 label should be removed from the non-complex labels (if it has been included) (True 
                    or False)
        """

        self.hdf5filename = hdf5filename
        self.transform = transform
        self.complexLabels = complexLabels
        self.removePhi10 = removePhi10

        with h5py.File(self.hdf5filename, "r") as flen:
            # Ranks refers to each parallel process used to save simulations to HDF5 file
            numRanks = flen["ronch dataset"].shape[0]
            # Note: this accuracy of the name of the below variable relies on all of the HDF5 file memory spaces 
            # being filled with valid data, e.g. no incomplete simulations
            ronchsPerRank = flen["ronch dataset"].shape[1]

            self.length = numRanks * ronchsPerRank

    def __len__(self):
        return self.length

    def open_hdf5(self):

        self.f = h5py.File(self.hdf5filename, "r")

        self.RandMags = self.f["random_mags dataset"]
        self.RandAngs = self.f["random_angs dataset"]
        self.ronchs = self.f["ronch dataset"]

        self.RandI = self.f["random_I dataset"]
        self.Randt = self.f["random_t dataset"]

    def __getitem__(self, idx):
        """idx is the single-number index referring to the item being got. Since, for each of self.RandMags, 
        self.RandAngs and self.ronchs, the first dimension is rank and only the second dimension is the item itself, 
        this method must take idx and calculate from it a corresponding rank and index within that rank itself. Here, 
        an aberration's magnitude and angle are converted to a single complex number."""

        if not hasattr(self, 'f'):
            self.open_hdf5()

        numRanks = self.ronchs.shape[0]
        itemsPerRank = self.ronchs.shape[1]

        rank = idx // itemsPerRank
        itemInRank = idx % itemsPerRank

        ronch = self.ronchs[rank, itemInRank]

        mags = self.RandMags[rank, itemInRank]
        angs = self.RandAngs[rank, itemInRank]

        # TODO: get rid of the below (see Google doc 16/02/22)
        if self.complexLabels:

            # Putting the aberrations into complex form
            # TODO: see if there is a function that does what you are doing below (i.e. that can take an array of moduli 
            # and an array of arguments and return an array of complex numbers) without you having to use a for loop

            # Array of aberrations in complex form, where each element will be for C10, C12, C21, and C23 respectively, each 
            # element being a complex number whose modulus is aberration magnitude/m and whose argument is aberration phi_n,m/rad
            complexArray = np.array([])

            for aber in range(len(mags)):
                # NOTE: cmath.rect() has an inherent error in it, for example, cmath.rect(1, cmath.pi/2) leads to 
                # 10**-19 + 1j rather than simply 1j.
                complexAber = cmath.rect(mags[aber], angs[aber])

                complexArray = np.append(complexArray, complexAber)

            # Okay, so somewhere in __getitem__, I put "if self.transform", as if there might not be a transform. However, I think there will always 
            # be ToTensor(), at least for what I will be doing for a while, so I am going to make sure that the label gets 
            # transformed to a tensor regardless.
            complexArray = torch.from_numpy(complexArray)

            # Okay, I am going to put this line of code here until I can find a better workaround for complex numbers. To 
            # make my network architecture work with complex numbers directly, I would have to put time in to do that, so 
            # for now I will just split complexArray into two tensors, one with its real parts and the other with its 
            # imaginary parts. Next I will create a new tensor, i.e. an 8-element vector whose first 4 elements are the real 
            # parts and whose latter 4 elements are the corresponding imaginary parts.
            realPart = torch.real(complexArray)
            imagPart = torch.imag(complexArray)

            # NOTE: here, complex array is a bit of a misnomer
            # NOTE: here, I have made dtype=torch.float32 because I believe dtype=torch.float64 is equivalent to 
            # torch.DoubleTensor, which MSELoss() in cnns/training.py doesn't seem to be accepting.
            labelsArray = torch.cat((realPart, imagPart)).to(dtype=torch.float32)

            # Decomment if you go back to using the magnitudes and angles themselves as labels, although will have to convert 
            # magnitude and angle array labels to torch Tensor like above
            # sample = {"ronchigram": ronch, "aberration magnitudes": mags, "aberration angles": angs}

            # sample = {"ronchigram": ronch, "aberrations": complexArray}

            # NOTE: here I am changing the return to look more like an MNIST return, since the model I am using doesn't seem 
            # to work well on a dictionary format, but it works on MNIST in My_CNNs/CNN_4.py. See Google Drive > 4th Year > 
            # CNN Stuff for more details.

        else:

            labelsArray = np.concatenate((mags, angs))
            
            if self.removePhi10:
                labelsArray = np.delete(labelsArray, 4)

            labelsArray = torch.from_numpy(labelsArray)


        # Certain torchvision.transform transforms, like ToTensor(), require numpy arrays to have 3 dimensions 
        # (H x W x C) rather than 2D (as of 5:01pm 08/01/22), hence the below. I assume here that if ronch.ndim == 2, 
        # the numpy array is of the form (H x W), as required. The below in that case makes numpy array have shape 
        # (H x W x C), later ToTensor() turns shape to (C x H x W) if applied. 
        if ronch.ndim == 2:
            ronch = np.expand_dims(ronch, 2)

        if self.transform:
            # The below is so that ToTensor() normalises the Ronchigram to between 0 and 1 inclusive; the resultant torch 
            # Tensor (after transformation by ToTensor()) will have dtype torch.FloatTensor
            ronch = ronch.astype(np.uint8)
            
            ronch = self.transform(ronch)

        sample = (ronch, labelsArray)

        return sample

    def getIt(self, idx):
        """Returns I (the quoted Ronchigram capture current/A) and t (Ronchigram acquisition time/s)"""

        if not hasattr(self, 'f'):
            self.open_hdf5()

        numRanks = self.ronchs.shape[0]
        itemsPerRank = self.ronchs.shape[1]

        rank = idx // itemsPerRank
        itemInRank = idx % itemsPerRank

        I = self.RandI[rank, itemInRank]
        t = self.Randt[rank, itemInRank]

        return (I, t)

    def __del__(self):

        if hasattr(self, 'f'):
            print("Closing HDF5 file...")
            self.f.close()
            print("The HDF5 file is closed.")

        else:
            print("The HDF5 file is closed.")

# From https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
def showBatch(batchedSample):
    """Show Ronchigram and print its aberrations for a batch of samples."""

    images_batch, labels_batch = batchedSample[0], batchedSample[1]

    # Decomment if desired to see
    print(labels_batch)

    batch_size = len(images_batch)
    im_size = images_batch[0].size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.title("Batch from dataloader")

# Inspired by https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
def getMeanAndStd(dataloader, reducedBatches=None, specificDevice=None):
    """Returns the mean and standard deviation of all Ronchigrams in dataloader. reducedBatches is the number of batches to 
    stop after if just testing out this function. Otherwise, don't pass an argument to it if want to really calculate 
    mean and std over every single batch.
    
    specificDevice: if None, runs the below on the default device; otherwise, runs the below on the device specified
    """

    sum, squaredSum, numBatches = 0, 0, 0
    
    # First index in enumerate(dataloader) is of course the index assigned to each iterable, the second index is the 
    # batch contained by the first index, and in this batch is a dictionary whose keys are "ronchigram" (whose value 
    # is a single tensor containing all Ronchigram tensors in batch) and "aberrations" (whose value is a single tensor 
    # containg all aberration tensors in batch)
    for iBatch, batch in enumerate(dataloader):
        print(f"Looking at batch at index {iBatch}...")

        # Mean over all tensor elements in batch
        batchedRonchs = batch[0]

        if specificDevice:
            batchedRonchs = convert_tensor(batchedRonchs, device=device, non_blocking=True)

        sum += torch.mean(batchedRonchs)
        squaredSum += torch.mean(batchedRonchs ** 2)
        numBatches += 1

        if iBatch + 1 == reducedBatches:
            break

    del batchedRonchs

    # Mean across all batches
    mean = sum / numBatches

    # std = sqrt(E(X^2) - (E[X])^2)
    std = (squaredSum / numBatches - mean ** 2) ** 0.5

    return mean, std

def getMeanAndStd2(ronchdset, trainingResolution, diagnosticBatchSize=4, diagnosticShuffle=True, batchesTested=32, specificDevice=None):
    """Takes a Ronchigram dataset and estimates its mean and std after transforms besides torchvision.transforms.Normalize() are applied. 
    TODO: if/when transforms used for training and testing are changed, modify diagnosticTransform below accordingly.

    ronchdset: total dataset of Ronchigrams, RonchigramDataset object

    trainingResolution: the size a Ronchigram should have before training begins, i.e. (trainingResolution x trainingResolution) pixels

    diagnosticBatchSize: size of batches to be iterated through in estimating the mean and std

    diagnosticShuffle: whether shuffling is done in the torch.utils.data.DataLoader object below; best to keep True for a less biased estimate of mean and std

    batchesTested: the number of batches of size diagnosticBatchSize to be sampled for estimation of mean and std

    device: if False, runs the below on the default device; otherwise, runs the below on the device specified
    """

    # Transform that is applied to the data before mean and std are calculated; should be calculated for transforms that are done before 
    # the Normalize() is applied
    diagnosticTransform = Compose([
        ToTensor(),
        Resize(trainingResolution, F2.InterpolationMode.BICUBIC)
    ])

    ronchdset.transform = diagnosticTransform

    diagnosticDataloader = DataLoader(ronchdset, batch_size=diagnosticBatchSize, shuffle=diagnosticShuffle, num_workers=0)

    mean, std = getMeanAndStd(diagnosticDataloader, batchesTested, specificDevice=specificDevice)

    # File for logging mean and std's calculated above. Log entries will include date and time of entry,  mean and std, 
    # number of batches and batch size, and torch seed.
    with open("/home/james/VSCode/DataLoading/MeanStdLog.txt", "a") as f:
        try:
            f.write(f"\n\n{scriptTime}")
            f.write(f"\nCalculated mean: {mean}\nCalculated std: {std}")
            f.write(f"\nMean and std calculated from {batchesTested} batches of size {diagnosticBatchSize}")
            f.write(f"\nShuffling was {diagnosticShuffle}; random module's seed and torch's seed were {torchSeed}")
        except:
            pass

    return mean, std

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

if __name__ == "__main__":

    # Seeding
    # 22 is arbitrary here
    seed = 22
    random.seed(seed)

    # Random seed or a fixed seed (defined above)
    torchReproducible = True

    if torchReproducible:
        torchSeed = seed
    else:
        torchSeed = torch.seed()

    torch.manual_seed(torchSeed)

    # GPU STUFF
    usingGPU = False

    if usingGPU:
        GPU = 0
        device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        GPU = torch.cuda.current_device()
        print(f"GPU: {GPU}")


    # Dataset instantiation

    ronchdset = RonchigramDataset("/media/rob/hdd1/james-gj/Simulations/16_02_22/Single_Aberrations.h5")
    ronchdset.complexLabels = False


    # Quick check of the numpy array plotting

    # NOTE: the below might look funny if the datatype of the numpy array is changed to np.uint8 in __getitem__ so that 
    # I could get ToTensor() to normalise the Ronchigrams to in between 0 and 1 inclusive
    plt.figure()
    show_data(ronchdset[450][0], ronchdset[450][1])
    plt.show()

    # Implementing a way to find the mean and std of the data for Normalize(). 
    # Since this relies on ToTensor() being done, I am going to create a new composed transform variable containing just 
    # ToTensor() and Resize(resolution, F2.InterpolationMode.BICUBIC). Going to exclude Normalize() of course because we 
    # are looking for the mean and std to pass to Normalize(), which should only act after the image has been converted to a 
    # torch Tensor with values between 0 and 1 inclusive and then resized to the desired resolution.

    # Image size must be 600 x 600 for EfficientNet-B7; be careful if you instead want to look at things for a different model 
    # of EfficientNet
    resolution = 600

    scriptTime = datetime.datetime.now()

    calculatedMean, calculatedStd = getMeanAndStd2(ronchdset=ronchdset, trainingResolution=resolution)


    # Applying transforms

    # trainTransform and testTransform both have toTensor() because both train and test data must be converted to torch 
    # Tensor for operations by torch; trainTransform and testTransform both have Resize(), with the same arguments, for 
    # consistency; trainTransform and testTransform both have Normalize(), with the same mean and std, for consistency.

    # Images plotted in tests below deviate from what the simulated Ronchigrams look like since matplotlib clips the negative 
    # array elements resulting from Normalize(). However, as long as Normalize is done with the same mean and std for both 
    # train and test data, the consistency should be fine. Anyway, images plotted below aren't exactly what the neural network 
    # "sees".

    # Image size must be 600 x 600 for EfficientNet-B7
    resolution = 600

    # TODO: try works if mean and std of data are being calculated earlier in the script; except assigns fixed values to them, 
    # preferably values found previously - going to develop that bit such that it changes depending on mean and std already 
    # found, and stored somewhere, since don't want to calculate mean and std for same data over and over again.
    try:
        mean = calculatedMean
        std = calculatedStd
    except:
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

    ronchdset.transform = trainTransform

    # testItem = ronchdset[50000][0]
    # print(testItem)
    # print(type(testItem))


    # Implementing torch.utils.data.DataLoader works on the above by adapting the third step, train and test transforms 
    # incorporated, and testing the dataloader

    dataloader = DataLoader(ronchdset, batch_size=4, shuffle=True, num_workers=0)


    # Applying transforms

    # trainTransform and testTransform both have toTensor() because both train and test data must be converted to torch 
    # Tensor for operations by torch; trainTransform and testTransform both have Resize(), with the same arguments, for 
    # consistency; trainTransform and testTransform both have Normalize(), with the same mean and std, for consistency.

    # Images plotted in tests below deviate from what the simulated Ronchigrams look like since matplotlib clips the negative 
    # array elements resulting from Normalize(). However, as long as Normalize is done with the same mean and std for both 
    # train and test data, the consistency should be fine. Anyway, images plotted below aren't exactly what the neural network 
    # "sees".

    # Image size must be 600 x 600 for EfficientNet-B7
    resolution = 600

    # TODO: try works if mean and std of data are being calculated earlier in the script; except assigns fixed values to them, 
    # preferably values found previously - going to develop that bit such that it changes depending on mean and std already 
    # found, and stored somewhere, since don't want to calculate mean and std for same data over and over again.
    try:
        mean = calculatedMean
        std = calculatedStd
    except:
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

    tempTrasnform = Compose([
        ToTensor()
    ])

    ronchdset.transform = trainTransform

    # testItem = ronchdset[50000][0]
    # print(testItem)
    # print(type(testItem))


    # Implementing torch.utils.data.DataLoader works on the above by adapting the third step, train and test transforms 
    # incorporated, and testing the dataloader

    dataloader = DataLoader(ronchdset, batch_size=4, shuffle=True, num_workers=0)

    testingDataLoader = True

    if testingDataLoader:
        for iBatch, batchedSample in enumerate(dataloader):
            print(iBatch, batchedSample[0].size(),
                    batchedSample[1].size())

            if iBatch == 3:
                plt.figure()

                showBatch(batchedSample)
                # print(batchedSample["aberrations"])
                plt.ioff()
                plt.show()

                break

    # Checking if random_split works by splitting ronchdset into train, eval and test
    # TODO: be careful because there are also dataloaders above, the memory they take up may be high, which is bad if they 
    # are unnecessary

    ronchdsetLength = len(ronchdset)

    trainLength = math.ceil(ronchdsetLength * 0.70)
    evalLength = math.ceil(ronchdsetLength * 0.15)
    testLength = ronchdsetLength - trainLength - evalLength

    trainSet, evalSet, testSet = random_split(dataset=ronchdset, lengths=[trainLength, evalLength, testLength], generator=torch.Generator().manual_seed(torchSeed))

    print(ronchdset.getIt(10))
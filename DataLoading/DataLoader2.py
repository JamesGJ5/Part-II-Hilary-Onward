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
import random

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
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
    magnitude/m being its modulus and the angle/rad being its argument. Currently, aberrations are C10, C12, C21, C23 and 
    C30 in Krivanek notation."""

    def __init__(self, hdf5filename: str, transform=None, complexLabels=False, 

                c10=False, c12=False, c21=False, c23=False, c30=False, c32=False, c34=False, c41=False, c43=False, 
                c45=False, c50=False, c52=False, c54=False, c56=False,

                phi10=False, phi12=False, phi21=False, phi23=False, phi30=False, phi32=False, phi34=False, phi41=False,
                phi43=False, phi45=False, phi50=False, phi52=False, phi54=False, phi56=False,

                c10scaling=1, c12scaling=1, c21scaling=1, c23scaling=1, c30scaling=1, c32scaling=1, c34scaling=1, 
                c41scaling=1, c43scaling=1, c45scaling=1, c50scaling=1, c52scaling=1, c54scaling=1, c56scaling=1,

                phi10scaling=1, phi12scaling=1, phi21scaling=1, phi23scaling=1, phi30scaling=1, phi32scaling=1, 
                phi34scaling=1, phi41scaling=1, phi43scaling=1, phi45scaling=1, phi50scaling=1, phi52scaling=1,
                phi54scaling=1, phi56scaling=1

                ):
        """Args:
                hdf5filename: path to the HDF5 file containing the data as mentioned in the comment under this class' definition
                
                transform (callable, optional): transforms being incroporated

                complexLabels: whether the labels will be in complex form or not

                cnm & phinm: whether or not each cnm and phinm should be included in the label returned, True or False

                # NOTE: cnmscaling & phinmscaling only apply when complexLabels == False
                cnmscaling & phinmscaling: how much to multiply each cnm and phi by, num
        """

        self.hdf5filename = hdf5filename
        self.transform = transform
        self.complexLabels = complexLabels

        cnm = (c10, c12, c21, c23, c30, c32, c34, c41, c43, c45, c50, c52, c54, c56)
        phinm = (phi10, phi12, phi21, phi23, phi30, phi32, phi34, phi41, phi43, phi45, phi50, phi52, phi54, phi56)

        self.cnmIndices = [i for i, x in enumerate(cnm) if x]
        self.phinmIndices = [i for i, x in enumerate(phinm) if x]

        self.cnmscaling = np.array([c10scaling, c12scaling, c21scaling, c23scaling, c30scaling, c32scaling, c34scaling,
                                    c41scaling, c43scaling, c45scaling, c50scaling, c52scaling, c54scaling, c56scaling])

        self.phinmscaling = np.array([phi10scaling, phi12scaling, phi21scaling, phi23scaling, phi30scaling, phi32scaling,
                                        phi34scaling, phi41scaling, phi43scaling, phi45scaling, phi50scaling, 
                                        phi52scaling, phi54scaling, phi56scaling])

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

        if not self.complexLabels:

            mags *= self.cnmscaling
            angs *= self.phinmscaling

            chosenMags = np.take(mags, self.cnmIndices)
            chosenAngs = np.take(angs, self.phinmIndices)

            labelsArray = np.concatenate((chosenMags, chosenAngs))

            labelsArray = torch.from_numpy(labelsArray)

        else:

            # If each cnm doesn't have its corresponding phinm present, can't form a complex number whose modulus is 
            # cnm/unit and whose argument is corresponding phinm
            assert self.cnmIndices == self.phinmIndices

            chosenMags = np.take(mags, self.cnmIndices)
            chosenAngs = np.take(angs, self.phinmIndices)

            complexArray = np.array([])

            for aber in range(len(chosenMags)):
                # NOTE: cmath.rect() has an inherent error in it, for example, cmath.rect(1, cmath.pi/2) leads to 
                # 10**-19 + 1j rather than simply 1j.

                complexAber = cmath.rect(chosenMags[aber], chosenAngs[aber])

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

        # NOTE: here I changed the return to look more like an MNIST return, since the model I am using doesn't seem 
        # to work well on a dictionary format, but it works on MNIST in My_CNNs/CNN_4.py. See electronic lab book 
        # entry "CNN Stuff 13/01/22, 14/01/22, 15/01/22, 16/01/22, 17/01/22 and 18/01/22" for more details.

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

    print("Running showBatch")

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

def getMeanAndStd2(ronchdset, trainingResolution, diagnosticBatchSize=4, diagnosticShuffle=True, batchesTested=32, specificDevice=None, apertureSize=512):
    """Takes a Ronchigram dataset and estimates its mean and std after transforms besides torchvision.transforms.Normalize() are applied. 
    TODO: if/when transforms used for training and testing are changed, modify diagnosticTransform below accordingly.

    ronchdset: total dataset of Ronchigrams, RonchigramDataset object

    trainingResolution: the size a Ronchigram should have before training begins, i.e. (trainingResolution x trainingResolution) pixels

    diagnosticBatchSize: size of batches to be iterated through in estimating the mean and std

    diagnosticShuffle: whether shuffling is done in the torch.utils.data.DataLoader object below; best to keep True for a less biased estimate of mean and std

    batchesTested: the number of batches of size diagnosticBatchSize to be sampled for estimation of mean and std

    device: if False, runs the below on the default device; otherwise, runs the below on the device specified

    apertureSize: radius of objective aperture in input data in pixels, num
    """

    # Transform that is applied to the data before mean and std are calculated; should be calculated for transforms that are done before 
    # the Normalize() is applied
    diagnosticTransform = Compose([
        ToTensor(),
        CenterCrop(np.sqrt(2) * apertureSize),
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
            f.write(f"\n{ronchdset.hdf5filename}")
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

    # SEEDING

    # 22 is arbitrary here
    seed = 22
    # random.seed(seed)

    # Random seed or a fixed seed (defined above)
    torchReproducible = True

    if torchReproducible:
        torchSeed = seed
    else:
        torchSeed = torch.seed()

    # torch.manual_seed(torchSeed)


    # GPU STUFF
    usingGPU = False

    if usingGPU:
        GPU = 0
        device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        GPU = torch.cuda.current_device()
        print(f"GPU: {GPU}")


    # DATASET INSTANTIATION

    ronchdset = RonchigramDataset("/media/rob/hdd1/james-gj/Simulations/forTraining/02_04_22/partiallyCorrectedSTEM_3.h5", 
    c10=True, c12=True, c21=True, c23=True, c30=True, c32=True, c34=True, c41=True, c43=True, c45=True, c50=True, 
    c52=True, c54=True, c56=True,
    phi10=True, phi12=True, phi21=True, phi23=True, phi30=True, phi32=True, phi34=True, phi41=True, phi43=True, 
    phi45=True, phi50=True, phi52=True, phi54=True, phi56=True)

    chosenIndices = [random.randint(0, len(ronchdset) - 1) for i in range(100)]
    # print(f"Chosen indices: {chosenIndices}")

    # print(f"Shape of Ronchigram in item at index {idx} of dataset: {ronchdset[idx][0].shape}")
    # print(f"Ronchigram array in item at index {idx} of dataset: {ronchdset[idx][0]}")
    # print(f"Label in item at index {idx} of dataset: {ronchdset[idx][1]}")
    # print(f"Number of items in dataset: {len(ronchdset)}")

    # sys.exit()

    # QUICK CHECK OF THE NUMPY ARRAY PLOTTING

    # NOTE: the below might look funny if the datatype of the numpy array is changed to np.uint8 in __getitem__ so that 
    # I could get ToTensor() to normalise the Ronchigrams to in between 0 and 1 inclusive
    plt.figure()

    for idx in chosenIndices:
        print(f"\nIndex {idx}")
        print(f"Label: {ronchdset[idx][1]}")

        show_data(ronchdset[idx][0], ronchdset[idx][1])
        plt.show()


    # ESTIMATING MEAN AND STD

    # Implementing a way to find the mean and std of the data for Normalize(). 
    # Since this relies on ToTensor() being done, I am going to create a new composed transform variable containing just 
    # ToTensor() and Resize(resolution, F2.InterpolationMode.BICUBIC). Going to exclude Normalize() of course because we 
    # are looking for the mean and std to pass to Normalize(), which should only act after the image has been converted to a 
    # torch Tensor with values between 0 and 1 inclusive and then resized to the desired resolution.

    # Image size must be 300 x 300 for EfficientNet-B3; be careful if you instead want to look at things for a different model 
    # of EfficientNet
    resolution = 300

    scriptTime = datetime.datetime.now()

    apertureSize = 1024 / 2 # Aperture radius in pixels

    estimateMeanStd = False

    if estimateMeanStd:

        calculatedMean, calculatedStd = getMeanAndStd2(ronchdset=ronchdset, trainingResolution=resolution, apertureSize=apertureSize)
        print(calculatedMean, calculatedStd)


    # APPLYING TRANSFORMS

    # trainTransform and testTransform both have toTensor() because both train and test data must be converted to torch 
    # Tensor for operations by torch; trainTransform and testTransform both have Resize(), with the same arguments, for 
    # consistency; trainTransform and testTransform both have Normalize(), with the same mean and std, for consistency.

    # Images plotted in tests below deviate from what the simulated Ronchigrams look like since matplotlib clips the negative 
    # array elements resulting from Normalize(). However, as long as Normalize is done with the same mean and std for both 
    # train and test data, the consistency should be fine. Anyway, images plotted below aren't exactly what the neural network 
    # "sees".

    # Image size must be 300 x 300 for EfficientNet-B3
    resolution = 300

    # TODO: try works if mean and std of data are being calculated earlier in the script; except assigns fixed values to them, 
    # preferably values found previously - going to develop that bit such that it changes depending on mean and std already 
    # found, and stored somewhere, since don't want to calculate mean and std for same data over and over again.
    try:
        mean = calculatedMean
        std = calculatedStd
    except:
        mean = 0.5006
        std = 0.2591

    trainTransform = Compose([
        ToTensor(),
        CenterCrop(np.sqrt(2) * apertureSize),
        Resize(resolution, F2.InterpolationMode.BICUBIC),
        Normalize(mean=[mean], std=[std])
    ])

    testTransform = Compose([
        ToTensor(),
        CenterCrop(np.sqrt(2) * apertureSize),
        Resize(resolution, F2.InterpolationMode.BICUBIC),
        Normalize(mean=[mean], std=[std])
    ])

    ronchdset.transform = trainTransform

    ronchSubset = Subset(ronchdset, chosenIndices)

    # Implementing torch.utils.data.DataLoader works on the above by adapting the third step, train and test transforms 
    # incorporated, and testing the dataloader

    dataloader = DataLoader(ronchSubset, batch_size=4, shuffle=False, num_workers=0)


    # TESTING THE DATALOADER

    testingDataLoader = True

    if testingDataLoader:
        for iBatch, batchedSample in enumerate(dataloader):
            print(iBatch, batchedSample[0].size(),
                    batchedSample[1].size())

            if iBatch == 0:
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

    print(f"I/A and t/s respectively: {ronchdset.getIt(10)}")
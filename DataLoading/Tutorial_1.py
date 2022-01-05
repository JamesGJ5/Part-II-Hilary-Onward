from __future__ import print_function, division
# TODO: this is a real module, have a look at it
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # Interactive mode

# Going to work with a face dataset for practice
# There are 68 differnt landmark points in each face image
# Reading the CSV and getting annotations in an (N, 2) array where N is number of landmarks

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

# NOTE: setting n to 65 means the below, up to a certain point, is just done for one row of the CSV file, i.e. for one image.
n = 65
img_name = landmarks_frame.iloc[n, 0]
# TODO: make sure you know what iloc does, and why the top row of CSV doesn't get selected
# The top line of the CSV file is just a description of the format of each line of it
# First element in the above slice is the row of the CSV file, each row corresponding to a different image
# The first element of each row (i.e. at index 0) is the image name
# So, the above makes a column vector of all the image names
landmarks = landmarks_frame.iloc[n, 1:]
# The rest of each row (i.e. after index 0) is the face landmark data, so the above selects all of that
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)
# Results in the aforementioned array in which there are 2 columns - in each row, there are two values for each landmark

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 landmarks: {}'.format(landmarks[:4]))
# TODO: understand why the first landmark is [32. 65.]

# Writing a helper function to show an image and its landmarks and use it to show a sample

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    # NOTE: the first argument is for the x-values (i.e. the first element of each part's tuple), whereas the second is 
    # for the y-values (i.e. the second element of each part's tuple). Plotting these against each other shows the position 
    # of each landmark.
    plt.pause(0.0001)    # Pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
# plt.show(block=True)

# Custom dataset should inherit Dataset (in torch.utils.data.Dataset) and override its __len__ and __getitem__ methods (see tutorial
# mentioned in README.txt for why)

# Will read the csv in __init__ but leave image reading to __get__item
# This is memory efficient because all the images are not stored in the memory at once but read as required.

# Sample of our dataset will be a dict {'image': image, 'landmarks': landmarks}
# dataset will take an optional argument transform so that any required processing can be applied on the sample. We will see the usefulness of transform in the next section.
# TODO: might skip the above in my own dataset since train_transform and test_transform are already established, but will see

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Instantiating this class and iterating through the data samples. Printing sizes of first 4 samples and showing their landmarks.

face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        # plt.show(block = True)
        break

# TRANSFORMS(WILL OF COURSE BE DIFFERENT TO THE WAY I DO MY OWN TRANSFORMS)
# Applying Rescale, RandomCrop (to crop from image randomly to make extra training data) and ToTensor
# Going to write the above as callable classes cf. simple functions so parameters of the transform need not 
# be passed everytime it's called. Just need to implement the __call__ method and, if required, __init__ method. Then use 
# transform like:

# tsfm = Transform(params)
# transformed_sample = tsfm(sample)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


# COMPOSING THE TRANSFORMS

# Rescaling shorter side of image to 256 then randomly cropping a square of size 224 from it, i.e. want to compose 
# Rescale and RandomCrop transforms. torchvision.transforms.Compose is a simple callable class which allows us to do 
# this.

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Applying each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()


# ITERATING THROUGH THE DATASET (before we realise that torch.utils.data.DataLoader is a good alternative)

transformed_dataset = FaceLandmarksDataset(
    csv_file='data/faces/face_landmarks.csv',
    root_dir='data/faces/',
    transform=transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor()
    ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())
    # N.b. in the link in README.txt, they apply a transformation in the previous stage including ToTensor, so their 
    # images are torch Tensors, so in the above print statements, they have .size() and what is printed is in torch 
    # format.

    if i == 3:
        break

# DataLoader is used so that we can batch the data, shuffle the data and load 
# the data in parallel using multiprocessing workers

# USING torch.utils.data.DataLoader

dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show an image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
        sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
        landmarks_batch[i, :, 1].numpy() + grid_border_size,
        s=10,
        marker='.',
        c='r'
        )

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
        sample_batched['landmarks'].size())

    # Observe 4th batch and stop
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
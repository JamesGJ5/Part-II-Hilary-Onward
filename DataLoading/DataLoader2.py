# "Let’s create a dataset class for our face landmarks dataset. We will read the csv in __init__ but leave the reading of images to __getitem__. This is memory efficient because all the images are not stored in the memory at once but read as required."
# TODO: think of similar ways to maximise efficiency

# TODO: make sure to implement a way to easily check if things are working as in teh tutorial and use it in incremental development.

# TODO: think about what transforms must be implemented in the data loading stage, i.e. which aren't sufficiently implemented
# in the training/testing pipeline already

# TODO: make sure to test not only the dataset itself but torch.utils.data.DataLoader too, perhaps in a similar way to 
# how the pytorch.org tutorial displays the batched data near the end

# Plan:

# First, make sure that the data can be read correctly at all from the HDF5 file, maybe display images and labels

# Second, create the RonchigramDataset class
# - Make sure the class definition is consistent with torch.utils.DataLoader by following 2:35 of 
# https://www.youtube.com/watch?v=zN49HdDxHi8:
# -- Make sure __init__ downloads, reads data etc.
# -- Make sure __getitem__ provides the "item" when an index of a single number is passed as an argument
# -- Make sure __len__ provides the size of the data, i.e. number of images I think
# -- TODO: DOUBLE CHECK THE LAST POINT ABOVE
# - Incorporate a file open/close or with method correctly
# - Incorporate a transforms option
# - After making each method, check it works

# Third, check that a bunch of samples can be plotted properly

# Fourth, apply transforms and repeat the third step

# Fifth, check that torch.utils.data.DataLoader works on the above by adapting the third step, 
# train and test transforms incorporated
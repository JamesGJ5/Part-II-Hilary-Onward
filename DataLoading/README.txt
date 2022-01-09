A directory I've made to learn how to create a Torch dataset and to actually do so with my own simulations.

Notable Contents:

    DataLoader2.py was made after I installed parallel HDF5 so I could save simulations from different 
    processes in the same file. I plan to implement a way to represent, in a single complex number, the stored aberration 
    magnitudes and associated phi_nm angles.

    MeanStdLog.txt is a log for storing mean and std values calculated in DataLoader2.py for the data in question. I am 
    using calculated values for the the mean and std arguments passed to the Normalize() transform.

Archived Contents:

    Tutorial_1.py was me following a tutorial from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    data/ just contains some of the resources used in Tutorial_1.py

    Data_Loader_1.py is an adapation to my own simulations, but the simulations that were stored before I had parallel 
    HDF5 enabled, and before implementing a way to represent, in a single complex number, the stored aberration magnitudes 
    and associated phi_nm angles. This file was made 12/12/2021 for CNN_6.py. It was made by adapting the procedures used 
    in the pytorch.org tutorial mentioned above, but adapting said procedures to my own data.

Extra information:

    See 2:35 of https://www.youtube.com/watch?v=zN49HdDxHi8 for necessities that my dataset class definition must have 
    for torch.utils.data.DataLoader to be able to work upon it correctly.

    torchvision.datasets.DataFolder and torchvision.datasets.VisionDataset at
    https://pytorch.org/vision/stable/datasets.html#base-classes-for-custom-datasets look applicable to my simulations.

    Unlike the above, torchvision.datasets.ImageFolder doesn't look applicable because it requires folder names to be class 
    names, and since "classes" for my data are continuous values, this would probably require too many folders.
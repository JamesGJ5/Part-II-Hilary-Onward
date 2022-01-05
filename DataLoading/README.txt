A directory I've made to learn how to create a Torch dataset and to actually do so with my own simulations.

Notable Contents:

    Tutorial_1.py was me following a tutorial from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    data/ just contains some of the resources used in Tutorial_1.py

    Data_Loader_1.py is an adapation to my own simulations, but the simulations that were stored before I had parallel 
    HDF5 enabled.

    Data_Loader_Parallel_HDF5.py was made after I installed parallel HDF5 so I could save simulations from different 
    processes in the same file.

Archived Contents:

    I have not yet started coding up the dataloading stage for the loading of simulations from a single HDF5 file 
    containing all the simulations and not just links to files which contain the simulations.

Extra information:


torchvision.datasets.DataFolder and torchvision.datasets.VisionDataset at
https://pytorch.org/vision/stable/datasets.html#base-classes-for-custom-datasets look applicable to my simulations.

Unlike the above, torchvision.datasets.ImageFolder doesn't look applicable because it requires folder names to be class 
names, and since "classes" for my data are continuous values, this would require too many folders.

Data_Loader_1.py is made 12/12/2021 for CNN_5.py. It is made by adapting the procedures used in the tutorial mentioned above, 
but adapting said procedures to my own data.
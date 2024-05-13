## Context

I made this repo for my MEng master's thesis at the University of Oxford. This project entailed training an EfficientNet CNN to perform multi-output regression on microscopic images, for which I built a machine learning pipeline with Git, VS Code, Python, NumPy, PyTorch, HDF5, Linux, OOP and Google Colab. The code here was written before I knew how to code well but I managed to use this code to generate presentable results for my report.

## Repo Description

Salient files/folders:

    currentPipelines/ (where I have my current model architecture, training pipeline and inferencer)
    DataLoading/ (where I have my current dataset class definition)
    Simulations/Parallel_HDF5_2.py (where I simulate Ronchigrams using Primary_Simulations_1.py in parallel)
    Simulations/Primary_Simulations_1.py (where one Ronchigram is simulated)
    Simulations/Simulations_Loading_1.py (where the the saving of simulations from Parallel_HDF5_2.py is checked)

On my account on the group computer: pytorch3copy or (if the pytorch3copy doesn't work) pyytorch3 are the usable 
conda environments.

Versions of necessary packages for the above files/folders:

    mamba (via "conda install mamba")
        - Better at resolving dependencies than conda is so use it in place of "conda" for conda commands

    HDF5 built against parallel HDF5 with MPI-supported h5py (via "conda install -c conda-forge "h5py>=2.9=mpi*"" or 
    "mamba install -c conda-forge "h5py>=2.9=mpi*"")
        - For running parallel_HDF5_2.py via parallel simulations by going to the command line, navigating to 
        Simulations/ and running "mpiexec -n <number of processes to use> python Parallel_HDF5_2.py"
    
    pytorch 3.9.7, cudatoolkit 11.3.1, torchvision 0.11.1
        - For files importing torch or torchvision or using cuda

    ignite 0.4.7 (via "conda install ignite -c pytorch" or "mamba install ignite -c pytorch")
        - For training.py (ignite has some useful metrics for training etc.)

    matplotlib
        - For training.py, DataLoader2.py and Primary_Simulations_1.py

    matplotlib-scalebar (from conda forge)
        - For making some example simulations with scalebards in Primary_Simulations_1.py

    scipy 1.7.1, scikit-image 0.18.3

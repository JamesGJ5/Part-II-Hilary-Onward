Directory made for tutorials I have followed on YouTube etc.

Contents:

    MPI_Tutorial_1.py, I believe, is the first file I created to check if MPI, as I had it installed, was working.

    Parallel_HDF5_1.py, I believe, is one I created closer to 04/01/2022 to test MPI for parallel after I actually set 
    it up properly, although I believe Parallel_HDF5_2.py and Simulation_Loading_1.py in VSCode/Simulations/ quickly 
    took over.

    Tutorial_1.py is where I have been trying out the occasional torch function, it seems, although it isn't specifically 
    for this purpose and seems quite flexible.

Extra information:

    This is a directory I have created to get started
    with VSCode and the Terminal in Linux.

    To learn to use VS Code specifically within Linux, 
    I am using https://www.youtube.com/watch?v=Y1fei1mzP7Q
    In fact, I'm going to assign Tutorial_1.py to this.

    This looks like a good cheat sheet for Linux 
    commands: https://www.guru99.com/linux-commands-cheat-sheet.html

    When you want to use conda, go to the Linux 
    Terminal and type bash. Then, use conda commands as 
    normal. To find the path of your interpreter when 
    selecting an interpreter that fits the conda env you are 
    looking for in VS Code for example, open Linux Terminal, 
    type bash, type conda activate "environment name", with no "", 
    then type "which python" with no "". Then come to VS Code, 
    type ctrl + shift + p, then type Python: Select Interpreter, then
    add this new path if it is not there, then select it.

    A tutorial on using the watch command in the Linux Terminal:
    https://linuxize.com/post/linux-watch-command/

    See here for some terminal commands like navigating up a directory, returning to the previous directory etc.:
    https://help.ubuntu.com/community/UsingTheTerminal

    For how to make a folder: https://www.cyberciti.biz/faq/how-to-make-a-folder-in-linux-or-unix/
    Just do "mkdir [new directory name]"

    2:51pm on 30/12/2021. I have just selected as my current interpeter one for the environment "ParallelHDF5", in 
    which I have installed 1.10.6.0a1 from https://pypi.org/project/h5pyp/1.10.6.0a1/#description, along with Python=3.8 
    for compatibility. In MPI_Tutorial_1.py I am first going to see if parallel HDF5 has been enabled successfully when 
    importing h5pyp and mpi4py. If not, I will think of a different approach. Note: there may be issues because the link is 
    to a legacy version of a PyPi package in Alpha, I picked the second-most recent version because I couldn't install the most 
    recent one on Python 3.10, although to be fair it was made after 3.9 came out and before 3.10; I could check if it works 
    with Python 3.9 but I am feeling lazy. In fact, I had better check if it does, I don't want to cut corners. Nope, that didn't 
    work, going back to version 1.10.6.0a1 and Python=3.8

    Actually, I ended up not using the project I tried to download above - see my Parallel HDF5 file in my Google Drive 
    for what I actually used, the one where I download, from conda-forge I think, h5py>=2.9=mpi, I think that is the 
    package name (basically h5py with mpi support turned on). However, it may be that the addional line of code:

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    in VSCode/Simulations/Parallel_HDF5_2.py may also have been needed, but I will check if things work without it.
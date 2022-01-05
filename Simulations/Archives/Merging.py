import os
import h5py

# Will practise by merging Single_Aberrations_0.h5 and Single_Aberrations_1.h5 (both from /rob/media/hdd1/james-gj/Ronchigrams/
# Simulations/08_12_2021).

# https://stackoverflow.com/questions/58187004/how-can-i-combine-multiple-h5-file/58293398#58293398

# With multiple datasets (which I have), extend process mentioned in above link using visititems() in h5py.
# All methods in process use glob() to find the HDF5 files used.

# Method 1: create external links. Seems best since it doesn't copy the data but provides access to the data in all 
# files via links in 1 file. This certainly bypasses any issues involved in raising memory used after copying data.

with h5py.File("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/links.h5", 'w') as h5fw:
    link_count = 0
    for h5name in os.listdir("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021"):
        if h5name == "links.h5":
            continue
        else:
            link_count += 1
            # h5fw['link' + str(link_count)] = h5py.ExternalLink(h5name, '/')
            h5fw[h5name + "_link"] = h5py.ExternalLink(h5name, '/')
            # Must still check how to use visititems()

# Don't worry about this line.
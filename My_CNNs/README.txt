Directory for my CNNs.

Notable Contents:
    
    NOTE: IN THE EXTRA INFORMATION SECTION, CNN_3.py, CNN_4.py AND CNN_5.py HAVE INITIALLY HAVE DIFFERENT MEANINGS 
    COMPARED TO THE BELOW. HOWEVER, IF YOU CONTINUE READING EXTRA INFORMATION, IT SEEMS I DELETED THOSE FILES, THEN 
    MADE NEW FILES OF THE SAME NAME.

    model1.py was made for storing the model for importation into other scripts.

    CNN_1.py, I believe, is where I first wrote code from the following Kaggle webpage to make a training/testing 
    pipeline for EfficientNet on CIFAR-10 data: https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook

    CNN_2.py is a file I then made to store relevant content from CNN_1.py, I think because I was having issues with the 
    memory taken up in the GPUs by CNN_1.py, either on my laptop, on the group computer or both. Eventually, I developed 
    CNN_2.py so that it worked in what it was made on Kaggle to do and properly stored the best weights from training. 
    Apparently, I ended up using the model path efficientNet_cifar10_9475.p.pt in the current My_CNNs/, but it seems not 
    here now, I probably deleted it since I wasn't interested in cifar-10. CNN_2.py has a bit shortly after the beginning 
    where one can choose what to compute.

    CNN_3.py was made to adapt CNN_2.py to recognise greyscale images. I adapted the network architecture to predict for 
    greyscale MNIST (recognising digits), although with only discrete label prediction. In CNN_2.py, the network 
    architecture was for recognising images with 3 colour channels, and all I had to do to make CNN_3.py work for 1 colour channel was change the first argument 
    of Conv2d in self.stem in the network architecture to 1. Additionally, changed the 3-tuple in train_transform for 
    the parameter "fill" to a single number, the first element of the 3-tuple that was. For both train_transform and 
    test_transform I changed the mean and std to just the first number of each.

    CNN_4.py, as presently constructed, was made to change predictions from CNN_3.py for MNIST digits from discrete to 
    continuous. One of the things I did was change the loss criterion from cross-entropy loss to MSELoss, a loss criterion 
    that permits prediction of continuous labels. I also introduced the metric MeanAbsoluteError from ignite.metrics to 
    evaluate the training stage, since it didn't really make sense to use top-1 and top-5 accuracy. I believe I wrote about 
    this in my lab book. I think I remember the training bits of this file working and predictions being continuous, 
    although the rest might be a bit dodgy. HOWEVER, IT LOOKS LIKE I HAVE WRITTEN MEANSQUAREDERROR WHERE I SHOULDN'T 
    HAVE IN HERE, I MUST AMEND THIS; I ALSO MIGHT HAVE TO TAKE OUT MENTIONS OF "ACCURACY" AS WELL, I THINK THESE APPLY 
    TO DISCRETE LABELS.

    CNN_5.py was made to show Chen that I had made progress, although there are definitely errors because I carried it 
    over from CNN_4.py for my own data but didn't test it. I MIGHT JUST CARRY OVER FROM CNN_4.py AGAIN INCREMENTALLY, 
    OR TWEAK CNN_5.py AS IT IS UNTIL IT WORKS.

    CNN_6.py (as of 04/01/2022) is a copy of CNN_5.py (unfixed) but with stuff from VSCode/DataLoading/Data_Loader_1.py 
    to demonstrate in vain my progress to Chen. I was going to attempt to just debug CNN_6.py and not CNN_5.py, but I will 
    probably debug the latter first to create less stress for myself, although I MIGHT EVEN MAKE SURE CNN_4.py WORKS FIRST.
    Anyway, I will have to change the dataloading bit of CNN_6.py after I change my dataloading file to work for a single HDF5 
    file then many, so not worth debugging CNN_6.py until after the new dataloading is put in.

    cifar-10-python.tar.gz seems to just be CIFAR-10 data for training the network, probably should archive them if 
    MNIST CNN_3.py and CNN_4.py files turn out to work.

    MNIST/raw is MNIST data, won't archive it yet since I want to see how well CNN_3.py and CNN_4.py work before moving on to 
    checking CNN_5.py and CNN_6.py which are for my own data.

    plans.txt, as of 04/01/2022, was last modified on 24/11/2022, which is the same day that CNN_4.py was created. That is 
    the second MNIST file, and I am going to be looking at the MNIST files, so plans.txt is fairly current and I will not yet 
    archive it.

Archived Contents:

    memory.py is a file I made for checking GPU memory attributes. It is useful but I will archive it because it isn't 
    such a notable file. Anyway, can still use it when I want to even if it is archived.

    test.py is just an arbitrary script for testing some torch functions. Will put it in Archives/ because can still access 
    it then but it is not an important file.

    EfficientNet_B7_Architecture_CNN_5 is just a save of the printed architecture of the EfficietnNet_B7 architecture used 
    in CNN_5.py, I assume. It is useful but I can print it whenever so I will archive it.

Extra information:

    To see free space, if nvidia-sim in Linux Terminal is 
    not enough: https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch

    Places in CNN_1.py where things are currently being stored on "device" (as of 20:24 on 22/11/2021 - I am trying to reduce 
    allocated space by the time I run update... function):

    Line 244 - to store the first instance of the EfficientNet model
    Line 294 - to store the transformed and unsqueezed giant panda image
    Line 299 - model is on the GPU, so y_pred gets stored there too, and it doesn't get unstored when the with statement ends, as
        I show in test.py
    Line 387 - to store the the images used in the classification step prior to tuning, although I think this is only one image. 
        Predictions should also get stored on GPU, as they are computed via model, which is on the GPU.
    Line 470 - tensors moved to GPU to be the images used
    Line 471 - predictions from images stored by Line 470
    Line 474 - I think these are the labels

    However, you would think the empty_cache statements sort this all out.

    CNN_2.py will contain relevant things from CNN_1.py. I will first try AMP to save memory.

    Note: when importing CNN_2.py into CNN_3.py to load the post-training parameters into the model, the aspect of CNN_2.py 
    where the Luke Melas weights are loaded becomes redundant. I should at some point put that into the main of CNN_2.py, 
    but I will not focus on that right now.
    Actually, I think I can put all that into if __name__ == "__main__": and just use from CNN_2 import (things I want) in 
    CNN_3.py.
    However, find a way to stop running everything in CNN_2.py when simply just importing things from it.

    I have decided to not use CNN_3.py, since I have realised that things were being saved all along. To be carefuly, I will 
    not delete it yet, but CNN_2.py should do the job for now. An issue in CNN_2.py, however, seems to be that log_path is 
    called such that it returns different datetime at different parts of the script, so two different folders get made. It 
    may not be this however. I am going to replace, for now, log_path with the exact path of the folder being looked for in 
    the checkpoints assignments section after the train.run bit of the script. Remember, log_path is of the format 
    f"/media/rob/hdd1/james-gj/finetune_efficientnet_cifar10/{exp_name}", where exp_name is datetime.

    I am making CNN_4.py to store the current version of CNN_2.py (21:05, 24/11/2021, after replacing log_path with exact 
    file name). I am doing this because I want to undo my changes in CNN_2.py, but CNN_4.py is correct. I just want to then 
    change CNN_2.py such that it is correct like CNN_4.py, but generalise it. I will, however, continue with testing etc. in 
    CNN_4.py.

    Note: having issue with CNN_4.py in that the file was deleted in the last inference (see the os bit with the rm command used), 
    and cannot find where it was copied to. May have to run training again, but at least I know how to get it all done. I must 
    improve CNN_2.py to generalise things for convenience. It is great to know that it works, however. Will sort out CNN_2.py 
    in morning quickly while using CNN_4.py for inspiration, then complete the secondary Ronchigram simulations and send to Chen.

    It seems maybe CNN_2.py wasn't making two log_path folders after all.

    Ok, I have set up CNN_2.py such that it correctly does things and stores the best model. 
    For further testing and finishing off what is done on the Kaggle webpage, I will now use 
    the model path efficientNet_cifar10_9475.p.pt, which is in the current directory. CNN_5.py will 
    be for the testing bit of this.

    In the end, CNN_2.py has all the features, with a section shortly after the beginning where one 
    can choose what to compute. I have deleted CNN_3.py, CNN_4.py and CNN_5.py. Going to quickly finish 
    implementing what is done in Kaggle webpage.

    Going to work on simulations now, since CNN_2.py is satisfactory.

    Now, I am going to see if I can find a practice dataset for images with labels that are continuous values. Going to look 
    in torch.

    CNN_3.py is where I am going to adapt the network to, first, a continuous-label practice dataset, then to my own data. 
    Going to adapt the network architecture such that it predicts out_features as a continuous distribution or something, 
    then make things work on MNIST handwritten digits, where the labels are numbers. Still may have to adjust loss type but 
    will first look at the webpage that says CrossEntropyLoss may be used.

    To note, the datasets that torch has access to which have numerical labels are 
    MNIST, QMNIST, Semeion, SHVN (although this seems segmentation-based) and maybe USPs.

    First, make things work on MNIST. Going to change c10classes variable to MNISTclasses variable. May need to change the 
    pretrained weights. But might do that later, really just want to check things work with numerical classes and not string 
    classes. Have changed the dataloading to MNIST. Note: got a value error 
    ValueError: The number of elements in 'fill' does not match the number of bands of the image (3 != 1), so changed the 
    argument of fill in the transforms to just the first element of the tuple from CNN_2.py. However, this led to an output 
    at some point with shape [1, 224, 224], which didn't match the broadcast shape [3, 224, 224]. Going to try to change 
    occurrences of 3 in the network itself to 1 - I will start by changing the 1st argument of Conv2d in self.stem of the 
    network to 1. May also need to change the list_channels argument. I am going to change both to 1. I have put a line 
    in in which this possibly changes, under where train_transform is written in CNN_2.py. An issue is that, when I change 
    these numbers directly, the pretrained weights don't work.

    OK, have just changed the first argument of nn.Conv2d (in self.stem) to 1. I think this is all that must be done. Got rid of the 
    bit where I load the pretrained weights. Changed, in train_transform, fill from a 3-tuple to the first number in that 
    3-tuple. Changed the mean and std in normalize of this train_transform to just the first number of each. Did the same 
    for test_transform. Things are training, however, I am slightly worried that the weights are just zero. I will print the 
    weights first to check.

    Note: still might have to put more thought into the numbers in the transforms.

    Ok, well it seems to work now for the greyscale images of MNIST. How well it works seems a bit sus, but then again, the 
    images of digits are fairly simple, re: recognising a 7 as opposed to a 3, for example.

    OK, now must change so that all predictions are continuous.

    Going to make CNN_4.py where I change things such that predictions are continuous.

    Must sort out The size of tensor a (10) must match the size of tensor b (125) at non-singleton dimension 1

    CNN_5.py has problems of its own, since it is an untested carry-over from CNN_4.py, adapted to my own data etc. What I 
    will do is create CNN_6.py where I copy over CNN_5.py and also the relevant bits from VSCode/DataLoading/Data_Loader_1.py. 
    I am being brave here but I will attempt to debug it all.
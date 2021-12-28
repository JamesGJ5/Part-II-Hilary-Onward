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

Code for CNN_1.py and CNN_2.py https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook

CNN_3.py is made 24/11/2021 to test the weights made in CNN_2.py on the testing cases etc. In CNN_2.py I have just 
changed the position of the active if __name__ == "__main__": statement so that I can import into CNN_2.py all the 
relevant things, i.e. the important bits before training. I will still have to put, after the training stage of 
CNN_3.py, a bit that saves the parameters, and then implement a bit in CNN_3.py that loads these parameters into the 
model. The old if __name__ == "__main__": statement in CNN_2.py is still evident, just commented out.


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

Just a line I am writing here to see if Git commands are working.
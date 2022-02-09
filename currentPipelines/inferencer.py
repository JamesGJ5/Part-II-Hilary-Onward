# Importations

import os
import torch
import model1
from ignite.metrics import MeanAbsoluteError, MeanSquaredError


# Navigating to the correct current working directory

os.chdir("/home/james/VSCode/currentPipelines")
print(f"Current working directory: {os.getcwd()}")


# Device configuration

GPU = 1
device = torch.device(f"cuda:{GPU}")
torch.cuda.set_device(GPU)
print(f"torch cuda current device: {torch.cuda.current_device()}")


# MODEL INSTANTIATION

efficientNetModel = "EfficientNet-B3"

# TODO: put the below in model1.py instead so you don't have to write it in every script that instantiates an EfficientNet() model
if efficientNetModel == "EfficientNet-B7":
    parameters = {"num_labels": 8, "width_coefficient": 2.0, "depth_coefficient": 3.1, "dropout_rate": 0.5}
    resolution = 600

elif efficientNetModel == "EfficientNet-B6":
    parameters = {"num_labels": 8, "width_coefficient": 1.8, "depth_coefficient": 2.6, "dropout_rate": 0.5}
    resolution = 528

elif efficientNetModel == "EfficientNet-B5":
    parameters = {"num_labels": 8, "width_coefficient": 1.6, "depth_coefficient": 2.2, "dropout_rate": 0.4}
    resolution = 456

elif efficientNetModel == "EfficientNet-B4":
    parameters = {"num_labels": 8, "width_coefficient": 1.4, "depth_coefficient": 1.8, "dropout_rate": 0.4}
    resolution = 380

elif efficientNetModel == "EfficientNet-B3":
    parameters = {"num_labels": 8, "width_coefficient": 1.2, "depth_coefficient": 1.4, "dropout_rate": 0.3}
    resolution = 300

elif efficientNetModel == "EfficientNet-B2":
    parameters = {"num_labels": 8, "width_coefficient": 1.1, "depth_coefficient": 1.2, "dropout_rate": 0.3}
    resolution = 260

elif efficientNetModel == "EfficientNet-B1":
    parameters = {"num_labels": 8, "width_coefficient": 1.0, "depth_coefficient": 1.1, "dropout_rate": 0.2}
    resolution = 240

elif efficientNetModel == "EfficientNet-B0":
    parameters = {"num_labels": 8, "width_coefficient": 1.0, "depth_coefficient": 1.0, "dropout_rate": 0.2}
    resolution = 224

model = model1.EfficientNet(num_labels=parameters["num_labels"], width_coefficient=parameters["width_coefficient"], 
                            depth_coefficient=parameters["depth_coefficient"], 
                            dropout_rate=parameters["dropout_rate"]).to(device)


# Loading weights

modelPath = "/media/rob/hdd2/james/training/fineTuneEfficientNet/20220208-161939/efficientNetBestReciprocalMSE_165727167543.3294"
model.load_state_dict(torch.load(modelPath))


# Choose metrics

# NOTE: in hindsight, this probably didn't need to be done before test-time augmentation and full inference on testLoader but it will come in handy eventually

metrics = {
    'MeanSquaredError': MeanSquaredError(),
    'MeanAbsoluteError': MeanAbsoluteError(),
}



# MUCH LATER: Test-time augmentation like in https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook
# MODERATELY LATER Running inferencer on entirety of testLoader
# SLIGHTLY LATER Appropriating inference to aberration magnitudes and phi_n,m angles, rather than the real and imaginary parts of the complex-number labels

# Adapting the image-plotting bit of the inference section of https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite/notebook to instead plot inferred 
# Ronchigrams and Ronchigrams from predicted labels side by side; print predicted labels alongside actual labels

    # Load the Ronchigrams to be inferred along with their real labels. Will probably just import testLoader from training.py since it has the relevant transforms 
    # applied already, hopefully that works out. Import evalLoader from training.py if necessary

from training.py import testLoader



    

    # Put model in eval() mode

    # Carry out evaluation of Ronchigrams to be inferred

    # Generate "predicted" Ronchigrams from predicted labels

    # Plot inferred Ronchigrams alongside predicted ones, maybe a 
    # row of the former above a row of the latter

    # Print actual labels for the above alongside predicted labels








# Print actual and predicted labels



# Inferencer after which a Ronchigram is produced from predicted labels



# MODERATELY LATER Maybe use Pandas to plot lots of inference results on mass
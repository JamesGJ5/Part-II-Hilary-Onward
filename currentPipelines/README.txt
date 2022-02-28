Uncluttered directory for CNN work. Use the Conda environment pytorch3 here (or its copy, pytorch3copy, if pytorch3 
hasn't been updated based on useful changes to pytorch3copy).

Notable Contents:
    
    model1.py will be where my model (and functions it relies on etc.) is stored.

    training.py will be my training pipeline.

    inferencer.py was created 1:21pm on 15/02/22 for a fresh script for carrying out inference on Ronchigrams and 
    plotting alongside them Ronchigrams calculated from labels predicted

    modelLogging logs model architectures used in each training run.

    modifyMAPE.py was made 9:25pm on 17/02/22 to modify the source code for MAPE loss criterion from 
    https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.metrics.MAPE.html to better suit my 
    needs. I needed to change the epsilon value (1e-8) first of all.

Archived Contents:

    inferencer.py was my initial inference pipeline.

How to Use:

    The Conda environment "pytorch3copy" works on this directory.
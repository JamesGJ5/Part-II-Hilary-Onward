import os
import numpy as np
import sys
import torch.nn as nn

exp_name = "20220118-011014"

log_path = f"/media/rob/hdd2/james/training/fineTuneEfficientNet/{exp_name}"
os.system(f"ls {log_path}")

# NOTE: out why print(next(os.walk(log_path)))[1] returns just an empty list - this 
# is because the ".pt" files really are just files, not directories (although their 
# icons make them look like directories). I think they are zip files.

# Takes all files in the directory log_path and its sub-directories
checkpoints = next(os.walk(log_path))[2]
print("\n" + str(checkpoints))

# Keeps only the files ending in ".pt"
checkpoints = sorted(filter(lambda f: f.endswith(".pt"), checkpoints))
print("\n" + str(checkpoints))

# TODO: automate calculation of the indices of the file names to take the 
# scores from below
scores = [eval(c[32:-3]) for c in checkpoints]
print("\nScores:", scores)

# TODO: check the following--I think bestEpoch is a misnomer, but for 
# some reason I have kept it called that like the Kaggle webpage I first 
# got this training pipeline from does. I think it actually just refers 
# to the best model but I could be wrong.
bestEpoch = np.argmax(scores)
print("\n" + str(bestEpoch))

if not checkpoints:
    print(f"\nThere are no weight files in {log_path}")

else:
    modelPath = f"{log_path}/efficientNetBestReciprocalMSE_{scores[bestEpoch]}"
    print("\nNew best model weights path:", modelPath)

    currentBestModelPath = os.path.join(log_path, checkpoints[bestEpoch])
    print("\n" + currentBestModelPath)

    os.system(f"cp -r {currentBestModelPath} {modelPath}")

removeOtherFiles = False

if removeOtherFiles:
    # Rather than remove everything in log_path, just want to remove the files 
    # that aren't the very best model weights. Don't want to remove everything 
    # in log_path because the very best model weights are stored there, so want 
    # to keep that.
    # TODO: as they arise, add other non-essential files to automatically delete, 
    # but not the loss curves, of course.
    os.system(f"rm {log_path}/best_model_test* {log_path}/events*")
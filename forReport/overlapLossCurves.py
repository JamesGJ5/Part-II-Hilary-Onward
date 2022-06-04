# File to overlap different loss curves (training loss running average, overall validation loss... on the same figure, 
# a different figure being used for each training run)

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import math

correctedSTEMRuns = [
    '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220525-194703',
    ]

figureTitles = {
    '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220525-194703': 'C3,0-Corrected STEM Training Run', 
    }

trainingRunDirs = correctedSTEMRuns

lossCurveFiles = [
    '/trainingLossRunningAverage.json', '/overallValidationLoss.json', 
    '/c12ValidationLoss.json', '/phi12ValidationLoss.json'
    ]

curveColours = {
    '/trainingLossRunningAverage.json': 'b', 
    '/overallValidationLoss.json': 'r', 
    '/c12ValidationLoss.json': 'c', 
    '/phi12ValidationLoss.json': 'y'
    }

curveLabels = {
    '/trainingLossRunningAverage.json': 'Training Loss Running Average', 
    '/overallValidationLoss.json': 'Overall Validation Loss',
    '/c12ValidationLoss.json': 'c1,2 Validation Loss',
    '/phi12ValidationLoss.json': '\u03A61,2 Validation Loss'
    }

assert lossCurveFiles[0] == '/trainingLossRunningAverage.json'

for directory in trainingRunDirs:

    # One plot per directory so start matplotlib things here, probably
    fig, ax = plt.subplots()

    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Mean Squared Error Loss Value")

    # Reusing title in saving figure, hence initialising it as its own variabale
    title=figureTitles[directory]
    ax.set_title(title)

    for lossCurve in lossCurveFiles:

        with open(directory + lossCurve, 'r') as f:

            # Firstly, must implement a way to read the data from the file

            data = json.load(f)

            # print(data)
            # print(len(data))
            
            # Initialising arrays for plotting
            iterationNumbers = np.array([])
            lossVals = np.array([])

            for dataPoint in data:
                # lossVals[i] is the loss value for the iteration at iterationNumbers[i]
                iterationNumbers = np.append(iterationNumbers, dataPoint[1])
                lossVals = np.append(lossVals, dataPoint[2])
            # print(iterationNumbers)
            # print(lossVals)

            if lossCurve == '/trainingLossRunningAverage.json':
                # For else statement (if/else statement position because, for a given directory, the loss curve 
                # '/trainingLossRunningAverage.json' should always be first)
                numberTrainingIterations = iterationNumbers[-1]

            else:
                iterationNumbers = iterationNumbers / np.amax(iterationNumbers) * numberTrainingIterations
                # print(iterationNumbers)

            ax.plot(iterationNumbers, lossVals, color=curveColours[lossCurve], label=curveLabels[lossCurve])

            # Each data point seems to have the format [time recorded, recording step (i.e. 1st recording, 2nd), 
            # recorded value]; each loaded json file seems to have the format [data point, data point ... data point]

            # Secondly, now that we have the data, must find a way to make the x-axis of each set of data such that the 
            # data lines up probably with data from other files

                # First of all, a good thing to do seems to be to remember that, while in the validation loss curves times 
                # corresponding to associated data points are different, the values "step" are the same

                # It might be a good idea to make the step values from /trainingLossRunningAverage.json the x-axis values 
                # and convert each step in the validation loss curve to the corresponding "step" in training loss running 
                # average, i.e. the final step of the epoch after which the validation is done.

                    # Firstly, must see if step in /trainingLossRunningAverage.json lines up with number of iterations. 
                    # The highest number of step shown in this file is 21870. Training was done for 10 epochs and there 
                    # were 85014 * 14/17 (so 70012) training images per epoch, with a batch size of 32, meaning 2187
                    # iterations per epoch and 21870 iterations overall. THEREFORE, I THINK IT IS SAFE TO ASSUME THAT, IN 
                    # /trainingLossRunningAverage.json, STEP NUMBER IS THE SAME AS ITERATION NUMBER

                # For each /trainingLossRunningAverage.json file, going to make the step number the x axis as a numpy array 
                # (taking the highest value of this array for later reference), the loss value the y axis value

                # For each validation loss file, going to add recording step to x labels numpy array (then multiply this array elementwise by 
                # the scalar that is the highest step number in /trainingLossRunningAverage.json); going to have y values 
                # be the loss value in each data point, of course

    ax.legend()
    plt.show()

    saveFig = input('Save figure? Input True or False: ')
    if saveFig:
        fig.figure.savefig(f'/media/rob/hdd1/james-gj/forReport/Partially-Corrected STEM/{title}')
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


    ax.legend()
    plt.show()

    saveFig = input('Save figure? Input True or False: ')
    if saveFig:
        fig.figure.savefig(f'/media/rob/hdd1/james-gj/forReport/_/{title}')
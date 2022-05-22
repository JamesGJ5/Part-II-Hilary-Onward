# File to stich together loss curves (training loss running average, overall validation loss... ) from consecutive 
# training runs all on the same figure

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import math

fiveHundredThousandRuns = [
    '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220413-104953', 
    '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220413-194108', 
    '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220419-101845', 
    '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220425-104947'
    ]

# figureTitle = '500,000 Ronchigrams'
figureTitle = '500,000 Ronchigrams Untruncated'

trainingRunDirs = fiveHundredThousandRuns

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


# One plot per directory so start matplotlib things here, probably
fig, ax = plt.subplots()

ax.set_xlabel("Iteration Number")
ax.set_ylabel("Mean Squared Error Loss Value")

# Reusing title in saving figure, hence initialising it as its own variabale
title=figureTitle
ax.set_title(title)

# 1. Here, initialise numpy arrays for iteration numbers of all four directories for: trainingLossRunningAverage; 
# overallValidationLoss; c12ValidationLoss; phi12ValidationLoss
# Decided to use the first directory's arrays as the result

trainingLossRunningAverageIterationNumbers = np.array([0])
overallValidationLossIterationNumbers = np.array([0])
c12ValidationLossIterationNumbers = np.array([0])
phi12ValidationLossIterationNumbers = np.array([0])

# 2. Here, initialise numpy arrays for loss values of all four directories for: trainingLossRunningAverage; 
# overallValidationLoss; c12ValidationLoss; phi12ValidationLoss

trainingLossRunningAverageVals = np.array([0])
overallValidationLossVals = np.array([0])
c12ValidationLossVals = np.array([0])
phi12ValidationLossVals = np.array([0])

# 3. Then, in the loops "for lossCurve in lossCurveFiles", generate iterationNumbers and lossVals as is being done so far, 
# allowing also the lining up of validation loss iterations with training loss running average iterations

# 4. Remove ax.plot, plt.show and figure saving from the for loops: this stuff will only be done after the for loops

# 5. Where ax.plot was, have four if statements, each say if lossCurve == '/trainingLossRunningAverage.json'... and in 
# each if statement, concatenate lossVals to the correct loss numpy array defined above; in the case of the 
# iterationNumbers arrays, increment each array by the maximum of the overall training loss iteration numbers array 
# (or 0 if it is currently empty) defined above, then concatenate to global array

# Do ax.plot, plt.show and figure saving with global arrays after the for loops end

for directory in trainingRunDirs:

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

            # Incrementing iterationNumbers for stitching to previous loss curves, if previous loss curves even exist 
            # yet
            if not directory == '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220413-104953':

                if lossCurve == '/trainingLossRunningAverage.json':

                    previousIterationNumbers = trainingLossRunningAverageIterationNumbers

                elif lossCurve == '/overallValidationLoss.json':

                    previousIterationNumbers = overallValidationLossIterationNumbers

                elif lossCurve == '/c12ValidationLoss.json':

                    previousIterationNumbers = c12ValidationLossIterationNumbers

                elif lossCurve == '/phi12ValidationLoss.json':

                    previousIterationNumbers = phi12ValidationLossIterationNumbers

                iterationNumbers = iterationNumbers + np.amax(previousIterationNumbers)
                # print(iterationNumbers[0])

            # Concatenating iterationNumbers and lossVals to global arrays
            if lossCurve == '/trainingLossRunningAverage.json':

                # if directory == '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220413-104953':

                #     trainingLossRunningAverageIterationNumbers = iterationNumbers
                #     trainingLossRunningAverageVals = lossVals

                # else:

                trainingLossRunningAverageIterationNumbers = np.concatenate((trainingLossRunningAverageIterationNumbers, 
                                                                            iterationNumbers))

                trainingLossRunningAverageVals = np.concatenate((trainingLossRunningAverageVals, lossVals))

                # print(trainingLossRunningAverageIterationNumbers[0])

            elif lossCurve == '/overallValidationLoss.json':

                # if directory == '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220413-104953':

                #     overallValidationLossIterationNumbers = iterationNumbers
                #     overallValidationLossVals = lossVals

                # else:

                overallValidationLossIterationNumbers = np.concatenate((overallValidationLossIterationNumbers, 
                                                                            iterationNumbers))

                overallValidationLossVals = np.concatenate((overallValidationLossVals, lossVals))

                # print(overallValidationLossIterationNumbers[0])

            elif lossCurve == '/c12ValidationLoss.json':

                # if directory == '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220413-104953':

                #     c12ValidationLossIterationNumbers = iterationNumbers
                #     c12ValidationLossVals = lossVals

                # else:

                c12ValidationLossIterationNumbers = np.concatenate((c12ValidationLossIterationNumbers, 
                                                                            iterationNumbers))

                c12ValidationLossVals = np.concatenate((c12ValidationLossVals, lossVals))

                # print(c12ValidationLossIterationNumbers[0])

            elif lossCurve == '/phi12ValidationLoss.json':

                # if directory == '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220413-104953':

                #     phi12ValidationLossIterationNumbers = iterationNumbers
                #     phi12ValidationLossVals = lossVals

                # else:

                phi12ValidationLossIterationNumbers = np.concatenate((phi12ValidationLossIterationNumbers, 
                                                                            iterationNumbers))

                phi12ValidationLossVals = np.concatenate((phi12ValidationLossVals, lossVals))

                # print(phi12ValidationLossIterationNumbers[0])

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


if title == '500,000 Ronchigrams Untruncated':

    ax.plot(trainingLossRunningAverageIterationNumbers[1:], trainingLossRunningAverageVals[1:], 
            color=curveColours['/trainingLossRunningAverage.json'], label=curveLabels['/trainingLossRunningAverage.json'])
    # print(trainingLossRunningAverageIterationNumbers[-1], trainingLossRunningAverageVals[-1], len(trainingLossRunningAverageIterationNumbers), len(trainingLossRunningAverageVals))

    ax.plot(overallValidationLossIterationNumbers[1:], overallValidationLossVals[1:], 
            color=curveColours['/overallValidationLoss.json'], label=curveLabels['/overallValidationLoss.json'])
    # print(overallValidationLossIterationNumbers[-1], overallValidationLossVals[-1], len(overallValidationLossIterationNumbers), len(overallValidationLossVals))

    # print(overallValidationLossIterationNumbers[0])

    ax.plot(c12ValidationLossIterationNumbers[1:], c12ValidationLossVals[1:], 
            color=curveColours['/c12ValidationLoss.json'], label=curveLabels['/c12ValidationLoss.json'])
    # print(c12ValidationLossIterationNumbers[-1], c12ValidationLossVals[-1], len(c12ValidationLossIterationNumbers), len(c12ValidationLossVals))

    ax.plot(phi12ValidationLossIterationNumbers[1:], phi12ValidationLossVals[1:], 
            color=curveColours['/phi12ValidationLoss.json'], label=curveLabels['/phi12ValidationLoss.json'])
    # print(phi12ValidationLossIterationNumbers[-1], phi12ValidationLossVals[-1], len(phi12ValidationLossIterationNumbers), len(phi12ValidationLossVals))


elif title == '500,000 Ronchigrams':

    ax.plot(trainingLossRunningAverageIterationNumbers[1001:], trainingLossRunningAverageVals[1001:], 
            color=curveColours['/trainingLossRunningAverage.json'], label=curveLabels['/trainingLossRunningAverage.json'])
    # print(trainingLossRunningAverageIterationNumbers[-1], trainingLossRunningAverageVals[-1], len(trainingLossRunningAverageIterationNumbers), len(trainingLossRunningAverageVals))

    ax.plot(overallValidationLossIterationNumbers[2:], overallValidationLossVals[2:], 
            color=curveColours['/overallValidationLoss.json'], label=curveLabels['/overallValidationLoss.json'])
    # print(overallValidationLossIterationNumbers[-1], overallValidationLossVals[-1], len(overallValidationLossIterationNumbers), len(overallValidationLossVals))

    # print(overallValidationLossIterationNumbers[0])

    ax.plot(c12ValidationLossIterationNumbers[2:], c12ValidationLossVals[2:], 
            color=curveColours['/c12ValidationLoss.json'], label=curveLabels['/c12ValidationLoss.json'])
    # print(c12ValidationLossIterationNumbers[-1], c12ValidationLossVals[-1], len(c12ValidationLossIterationNumbers), len(c12ValidationLossVals))

    ax.plot(phi12ValidationLossIterationNumbers[2:], phi12ValidationLossVals[2:], 
            color=curveColours['/phi12ValidationLoss.json'], label=curveLabels['/phi12ValidationLoss.json'])
    # print(phi12ValidationLossIterationNumbers[-1], phi12ValidationLossVals[-1], len(phi12ValidationLossIterationNumbers), len(phi12ValidationLossVals))

ax.legend()
plt.show()

saveFig = input('Save figure? Input True or False: ')
if saveFig:
    fig.figure.savefig(f'/media/rob/hdd1/james-gj/forReport/Partially-Corrected STEM/500k Runs/{figureTitle}')
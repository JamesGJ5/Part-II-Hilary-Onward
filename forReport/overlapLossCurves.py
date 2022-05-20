import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import math

simdim100mrad = '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220509-180434'
simdim50mrad = '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220509-081430'
simdim100mradTo50mrad = '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220508-211133'

figureTitles = {
    '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220509-180434': '100mrad', 
    '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220509-081430': '50mrad',
    '/media/rob/hdd2/james/training/fineTuneEfficientNet/20220508-211133': '100mrad Cropped to 50mrad'
    }

trainingRunDirs = [simdim100mrad, simdim50mrad, simdim100mradTo50mrad]

lossCurveFiles = [
    '/trainingLossRunningAverage.json', '/overallValidationLoss.json', '/c23ValidationLoss.json', 
    '/phi23ValidationLoss.json'
    ]

curveColours = {
    '/trainingLossRunningAverage.json': 'b', 
    '/overallValidationLoss.json': 'r',
    '/c23ValidationLoss.json': 'c',
    '/phi23ValidationLoss.json': 'y'
    }

curveLabels = {
    '/trainingLossRunningAverage.json': 'Training Loss Running Average', 
    '/overallValidationLoss.json': 'Overall Validation Loss',
    '/c23ValidationLoss.json': 'c2,3 Validation Loss',
    '/phi23ValidationLoss.json': '\u03A62,3 Validation Loss'
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
        fig.figure.savefig(f'/media/rob/hdd1/james-gj/forReport/Elementary Ronchigram Training Runs/Loss Curves/{title}')
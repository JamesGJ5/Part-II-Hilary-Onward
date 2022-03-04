from pytorch_forecasting.metrics import MultiHorizonMetric
import torch
import sys
import torch.nn as nn


# Loss functions

class modifiedMAPE(MultiHorizonMetric):
    """
    Mean absolute percentage error (in %, unlike pytorch_forecasting.metrics.MAPE where a fraction is computed instead). 
    Increments elements of y (label) equal to zero by a chosen amount, unlike for pytorch_forecasting.metrics.MAPE, 
    where every y is incremented by 1e-8.

    Defined as ``(y - y_pred).abs() / y.abs()`` for elements of y that are nonzero, but the denominator is augmented 
    for elemnents of y that are zero.
    """

    def loss(self, y_pred, target):
        # TODO: make epsilon (10**-13) an array that has different values for each element, according to what makes an element 
        # negligible (see Google doc 16/02/22)

        # NOTE: divided by 10**7 because this led to modifiedMAPE's handler to not have to clip the loss value to prevent
        # gradients that were too large, at least it seemed to work at 12pm on 18/02/22. If y_pred labels are made 
        # larger, however, it may be that the 10**7 is no longer sufficient.
        loss = (self.to_prediction(y_pred) - target).abs() / (target.abs() + 1e-13) / 10**7

        return loss

def myMAFE(output, target):
    """
    Mean absolute fraction error. 
    Increments elements of y (label) equal to zero by a chosen amount, unlike for pytorch_forecasting.metrics.MAPE, 
    where every y is incremented by 1e-8.

    Defined as ``(y - y_pred).abs() / y.abs()`` for elements of y that are nonzero, but the denominator is augmented 
    for elemnents of y that are zero.
    """

    # TODO: make epsilon (10**-13) an array that has different values for each element, according to what makes an element 
    # negligible (see Google doc 16/02/22)
    loss = torch.mean((output - target).abs() / (target.abs() + 10**-13))

    # NOTE: because the loss is on the CPU (I think) it is hard to simply use the if statement below to clip losses 
    # greater than 10**9 to 10**9, so I had to use the below method.
    a = loss < 10**9
    b = loss >= 10**9

    loss = loss * a + b * 10**9

    # if loss.size() != torch.Size([]):
    #     if loss > 10**9:
    #         loss[0] = 10**9

    return loss

def myMAFE2(output, target):
    """
    Mean absolute fraction error. 
    Increments elements of y (label) equal to zero by a chosen amount, unlike for pytorch_forecasting.metrics.MAPE, 
    where every y is incremented by 1e-8.

    Aberration angles associated with magnitudes that are negligible are discarded from mean calculation.

    Defined as ``(y - y_pred).abs() / y.abs()`` for elements of y that are nonzero, but the denominator is augmented 
    for elemnents of y that are zero.
    """

    # TODO: make epsilon (10**-13) an array that has different values for each element, according to what makes an element 
    # negligible (see Google doc 16/02/22)
    loss = (output - target).abs() / (target.abs() + 10**-13)

    # NOTE: because the loss is on the CPU (I think) it is hard to simpyl use the if statement below to clip losses 
    # greater than 10**9 to 10**9, so I had to use the below method.
    a = loss < 10**9
    b = loss >= 10**9

    loss = loss * a + b * 10**9

    # if loss.size() != torch.Size([]):
    #     if loss > 10**9:
    #         loss[0] = 10**9

    return loss


# Reductions

torchCriterion = nn.MSELoss(reduction="none")

def neglectNegligiblePhi(y_pred, y):
    """
    Implements torch.nn.MSELoss(reduction="none"), and then applies a reduction to 
    it (to result in a single loss value for backpropagation and gradient descent) that takes a mean over the output 
    vector for a given sample but neglects in the calculation of this mean the aberration angles whose corresponding 
    magnitudes are negligible. The rationale for this is that such angles might not be so discernable in this case, so 
    training the network to recognise such angles might bias it.

    y_pred: the batch of labels of the Ronchigrams predicted by the network
    y: the batch of actual labels of the Ronchigrams, from which undiscernable aberration angles can be identified
    unreducedLoss: unreduced output of loss function
    """

    # lossOutput, currently, is batchSize number of rows in which each row looks like: c10, c12, c21, c23, phi12, phi21, 
    # phi23. Going to neglect aberration angles corresponding to aberration magnitudes that are negligible. In the case 
    # of the simulations /media/rob/hdd1/james-gj/Simulations/forTraining/01_03_22/singleAberrations.h5, these are the 
    # three aberrations with the smallest magnitudes, since in simulations, magnitudes that were divisions of the 
    # predominant magnitude will be negligible and there is little chance (I think) of the aberration chosen to be the 
    # predominant one not being greater than the other two, unless said predominant magnitude is zero.

    # So, identify the largest of the first 4 elements of a row in y (I think it would be a row, at least), then 
    # take the mean over all 4 of corresponding elements as well as that for the only aberration angle corresponding to said predominant 
    # magnitude. If predominant magnitude is at index 0, take mean over 0, 1, 2 and 3 of lossOutput row; if 1, take mean over 0, 1, 2, 3 
    # and 4; if 2, take mean over 0, 1, 2, 3 and 5; if 3, take mean over 0, 1, 2, 3 and 6. So, take mean over 0, 1, 2, 3 
    # and argmax (out of 0, 1, 2 and 3) + 3 if argmax > 0.
    
    unredLoss = torchCriterion(y_pred, y)

    redLoss = torch.empty((len(y), 1))

    for rowIdx, yRow in enumerate(y):

        unredLossRow = unredLoss[rowIdx]

        bigMagIdx = torch.argmax(yRow[:4])

        if bigMagIdx == 0:

            unredLossRow = unredLossRow[:4]

        else:

            unredLossRow = torch.cat((unredLossRow[:4], unredLossRow[bigMagIdx + 3].view(1)))

        # print(unredLossRow)

        redLoss[rowIdx] = torch.mean(unredLossRow)

    return torch.mean(redLoss)

# Just code I left here to test the function neglectNegligiblePhi
# y = torch.tensor([[10, 1, 1, 1, 0, 0, 0], [1, 10, 1, 1, 0, 0, 0]])
# unredLoss = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])

# neglectNegligiblePhi(y, unredLoss)
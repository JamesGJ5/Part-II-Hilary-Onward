from pytorch_forecasting.metrics import MultiHorizonMetric
import torch
import sys

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

def myMAPE(output, target):
    """
    Mean absolute percentage error (in %, unlike pytorch_forecasting.metrics.MAPE where a fraction is computed instead). 
    Increments elements of y (label) equal to zero by a chosen amount, unlike for pytorch_forecasting.metrics.MAPE, 
    where every y is incremented by 1e-8.

    Defined as ``(y - y_pred).abs() / y.abs()`` for elements of y that are nonzero, but the denominator is augmented 
    for elemnents of y that are zero.
    """
    # TODO: make epsilon (10**-13) an array that has different values for each element, according to what makes an element 
    # negligible (see Google doc 16/02/22)
    loss = torch.mean((output - target).abs() / (target.abs() + 10**-13)) / 10**7

    # NOTE: because the loss is on the CPU (I think) it is hard to simpyl use the if statement below to clip losses 
    # greater than 10**9 to 10**9, so I had to use the below method.
    a = loss < 10**9
    b = loss >= 10**9

    loss = loss * a + b * 10**9

    # if loss.size() != torch.Size([]):
    #     if loss > 10**9:
    #         loss[0] = 10**9

    return loss
from pytorch_forecasting.metrics import MultiHorizonMetric

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
        loss = (self.to_prediction(y_pred) - target).abs() / (((target == 0) * 10**-13 + target).abs()) * 100
        return loss
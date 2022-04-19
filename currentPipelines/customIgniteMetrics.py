from typing import Callable, Union

import torch

from ignite.metrics import EpochMetric

def computeRMSPE(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    """Computes the root mean squared PERCENTAGE error."""
    sqErs = torch.square((y.view_as(y_pred) - y_pred) / y.view_as(y_pred))
    RMSFE = torch.sqrt(torch.mean(sqErs))   # The 'F' in RMSFE stands for fractional

    return 100.0 * RMSFE.item()

class RMSPercentageError(EpochMetric):
    """Calculates the Root Mean Squared Percentage Error.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)` and of type `float32`.

    .. warning::

        Current implementation stores all input data (output and target) as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: optional device specification for internal storage.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.
    """

    def __init__(
        self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = torch.device("cpu")
    ):
        super(RMSPercentageError, self).__init__(
            computeRMSPE, output_transform=output_transform, device=device
        )
from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class BaseLoss(ABC):
    """Abstract base class for defining loss functions used in neural
    networks.
    """

    def __init__(self) -> None:
        self.actual: NDArray
        self.predicted: NDArray
        self.input_gradient: NDArray

    def feed_forward(self, actual: NDArray, predicted: NDArray) -> float:
        """Passes the actual and predicted values forward and returns the
        loss value.
        """
        self.actual = actual
        self.predicted = predicted

        loss_value = self._apply()
        return loss_value

    def propagate_backwards(self) -> NDArray:
        """Passes the input gradient backward and returns the computed
        input gradient.
        """
        self.input_gradient = self._compute_gradient()
        return self.input_gradient

    @abstractmethod
    def _apply(self) -> Any:
        """Abstract method to apply the loss function."""
        message = "Every loss function must implement `_apply()` method"
        raise NotImplementedError(message)

    @abstractmethod
    def _compute_gradient(self) -> Any:
        """Abstract method to compute the gradient of the loss function."""
        message = (
            "Every loss function must implement `_compute_gradient()` method"
        )
        raise NotImplementedError(message)

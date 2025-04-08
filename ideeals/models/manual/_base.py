from abc import ABC, abstractmethod
from logging import basicConfig, INFO
from typing import Any

from numpy.typing import NDArray


class BaseManualModel(ABC):
    """Base machine learning model class."""

    def __init__(self) -> None:
        self.x_train: NDArray
        self.y_train: NDArray
        basicConfig(format="Model: %(message)s", level=INFO)

    @abstractmethod
    def fit(self, x_train: NDArray, y_train: NDArray, *args, **kwargs) -> None:
        """The data passed to this method would be copied and used as
        NumPy :class:`ndarray`.
        """
        self.x_train, self.y_train = x_train, y_train

    @abstractmethod
    def predict(self, x_test: NDArray) -> Any:
        message = "Every model should implement the `predict()` method"
        raise NotImplementedError(message)

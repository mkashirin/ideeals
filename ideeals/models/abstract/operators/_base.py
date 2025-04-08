from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class BaseOperator(ABC):
    """Abstract base class for operators that process input data."""

    def __init__(self) -> None:
        self.input_: NDArray
        self.output_: NDArray
        self.input_gradient: NDArray

    def feed_forward(self, input_: NDArray) -> NDArray:
        """Perform the forward pass and return the output."""
        self.input_ = input_
        self.output_ = self._apply()
        return self.output_

    def propagate_backwards(self, output_gradient: NDArray) -> NDArray:
        """Perform the backward pass and return the input gradient."""
        self.input_gradient = self._compute_gradient(output_gradient)
        return self.input_gradient

    @abstractmethod
    def _apply(self) -> Any:
        """Abstract method to apply the operator."""
        message = "Any Operator must implement `_apply()` method"
        raise NotImplementedError(message)

    @abstractmethod
    def _compute_gradient(self, output_gradient: NDArray) -> Any:
        """Abstract method to compute the gradient of the operator."""
        message = "Any Operator must implement `_gradient()` method"
        raise NotImplementedError(message)


class ParameterizedOperator(BaseOperator):
    """Abstract base class for parameterized operators."""

    def __init__(self, parameter: NDArray) -> None:
        super().__init__()
        self.parameter = parameter
        self.parameterized_gradient: NDArray

    def propagate_backwards(self, output_gradient: NDArray) -> Any:
        """Perform the backward pass and return the input gradient."""
        self.input_gradient = self._compute_gradient(output_gradient)
        self.parameterized_gradient = self._compute_parameterized_gradient(
            output_gradient
        )
        return self.input_gradient

    @abstractmethod
    def _compute_gradient(self, output_gradient: NDArray) -> Any:
        """Abstract method to compute the gradient of the operator."""
        message = "Any ParameterizedOperator must implement `_apply()` method"
        raise NotImplementedError(message)

    @abstractmethod
    def _compute_parameterized_gradient(self, output_gradient: NDArray) -> Any:
        """Abstract method to compute the parameterized gradient of the
        operator.
        """
        message = (
            "Any ParameterizedOperator must implement"
            "`_parameterized_gradient()` method"
        )
        raise NotImplementedError(message)

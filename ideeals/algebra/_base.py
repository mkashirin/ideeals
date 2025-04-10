from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


class BaseSolver(ABC):
    """Abstract base solver class to be inherited by other solvers."""

    def __init__(self) -> None:
        self.coefficient_matrix: NDArray
        self.scalars_vector: NDArray
        self.augmented_matrix: NDArray
        self.solution_vector: NDArray

    @abstractmethod
    def fit(
        self, coefficient_matrix: NDArray, scalars_vector: NDArray, **kwargs
    ) -> None:
        """Abstract method for fitting the solver and validate the inputs."""
        self.coefficient_matrix, self.scalars_vector = (
            coefficient_matrix,
            scalars_vector,
        )
        self.augmented_matrix = np.concatenate(
            (self.coefficient_matrix, self.scalars_vector), axis=1
        )

    @abstractmethod
    def solve(self, *, round_to: Optional[int] = None) -> Any:
        """Abstract method for solving the system of linear equations."""
        message = "Every solver must implement the `solve()` method"
        raise NotImplementedError(message)

    def _apply_partial_pivoting(self, iteration: int, rows: int) -> NDArray:
        """Perform partial pivoting to avoid division by zero."""
        for row in range(iteration + 1, rows):
            if np.abs(self.augmented_matrix[iteration, iteration]) < np.abs(
                self.augmented_matrix[row, iteration]
            ):
                self.augmented_matrix[[row, iteration]] = (
                    self.augmented_matrix[[iteration, row]]
                )

        if not self.augmented_matrix[iteration, iteration]:
            message = "Given system of linear equations is degenerate"
            raise ZeroDivisionError(message)
        return self.augmented_matrix

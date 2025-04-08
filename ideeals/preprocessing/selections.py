from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .._typing import Selections


def tokenize_probabilities(matrix: NDArray) -> NDArray:
    """Tokenize the probabilities of each outcome in the output layer.

    :param matrix: Probabilities matrix, filled with values ranging
    from 0 and 1.
        :type matrix: :class:`NDArray`

    :returns: Binary matrix, where columns represent classes and rows
    represent samples.
        :rtype: :class:`NDArray`
    """
    binary_matrix = np.zeros_like(matrix)
    max_indices = np.argmax(matrix, axis=1)
    rows = np.arange(matrix.shape[0])
    binary_matrix[rows, max_indices] = 1
    return binary_matrix


def revert_matrix(matrix: NDArray) -> NDArray:
    """Revert the target binary matrix to original vector format.

    :param matrix: The binary matrix to revert.
        :type matrix: :class:`NDArray`

    :returns: Vector formatted from the binary matrix passed.
        :rtype: :class:`NDArray`
    """
    return np.argmax(matrix, axis=1).reshape(-1, 1)


def transform_binary(vector: NDArray) -> NDArray:
    """Convert the target vector to binary matrix.

    :param vector: Target vector, which is to be transformed.
        :type vector: :class:`NDArray`

    :returns: Binary matrix, where columns represent classes and rows
    represent samples.
        :rtype: :class:`NDArray`
    """
    n_values = vector.shape[0]
    max_value = np.max(vector)
    binary_matrix = np.zeros((n_values, max_value + 1))
    for i, value in enumerate(vector):
        binary_matrix[i, value] = 1

    return binary_matrix


def reshape_channel_images(
    images: NDArray, n_channels: int, *, image_height: int, image_width: int
) -> NDArray:
    """Reshape channel images stored as arrays of pixels to be properly
    consumed by the convolutional neural nets.

    :param images: Array of images to be reshaped.
        :type images: :class:`NDArray`
    :param n_channels: Number of channels of the single image.
        :type n_channels: :class:`int`

    :keyword image_height: Height of the single image in pixels.
        :type image_height: :class:`int`
    :keyword image_width: Width of the single image in pixels.
        :type image_width: :class:`int`

    :returns: Array of images reshaped in a specified way.
        :rtype: :class:`NDArray`
    """
    reshaped = images.reshape(-1, n_channels, image_height, image_width)
    return reshaped


def normalize_data(*, to_normalize: NDArray, std_from: NDArray) -> NDArray:
    """Normalize the data using the standard deviation from another data.

    :keyword to_normalize: Array of data to be normalized.
        :type to_normalize: :class:`NDArray`
    :keyword std_from: Array to be taken standard deviation from.
        :type std_from: :class:`NDArray`

    :returns: Normalized array.
        :rtype: :class:`NDArray`
    """

    normalized = to_normalize / np.nanstd(std_from)
    return normalized


class DataSplitter:
    """Data splitting interface, which allows you to separate your data
    on train, validation and test selections.
    """

    def __init__(
        self, permute: bool = False, random_seed: Optional[int] = None
    ):
        """Set params for the splitting.

        :param permute: Defines whether data will be permuted before
        split operation or not.
            :type permute: :class:`bool`
        :param random_seed: Random seed that will be applied during
        the process.
            :type random_seed: :class:`Optional[int]`
        """
        self.random_seed = random_seed
        self.permute = permute
        self._selections: List[NDArray]

    def split_data(
        self,
        x: NDArray,
        y: NDArray,
        *,
        test_size: float,
        valid_size: Optional[float] = None,
    ) -> Selections:
        """Split the data on train, validation and test selections.

        :param x: Features data, that would be split on train and
        test selections.
            :type x: :class:`NDArray`
        :param y: Target data, that would be split on train and
        test selections.
            :type y: :class:`NDArray`
        :param test_size: Percentage of data that will be allocated for the
        test selection.
            :type test_size: :class:`float`

        :keyword valid_size: Percentage of data that will be allocated for the
        validation selection.
            :type valid_size: :class:`Optional[float]`

        :returns: Tuple of selections split according to specified params.
            :rtype: :class:`Selections`
        """
        if self.random_seed:
            np.random.seed(self.random_seed)
        if self.permute:
            permutation = np.random.permutation(x.shape[0])
            x, y = x[permutation], y[permutation]

        self._set_standard(x, y, test_size)
        if valid_size:
            test_length = self._selections[1].shape[0]
            self._add_valid(test_length, x, y, valid_size)

        selections: Selections = tuple(
            self._selections  # pyright: ignore[reportAssignmentType]
        )
        return selections

    def _set_standard(self, x: NDArray, y: NDArray, test_size: float) -> None:
        train_test_index = int(x.shape[0] * test_size)

        (x_train, x_test), (y_train, y_test) = (
            (x[train_test_index:], x[:train_test_index]),
            (y[train_test_index:], y[:train_test_index]),
        )
        self._selections = [x_train, x_test, y_train, y_test]

    def _add_valid(
        self, test_length: int, x: NDArray, y: NDArray, valid_size: float
    ) -> None:
        test_valid_index = int(test_length * valid_size)

        self._selections[1], self._selections[3] = (
            self._selections[1][test_valid_index:],
            self._selections[3][test_valid_index:],
        )
        self._selections.insert(1, x[:test_valid_index])
        self._selections.insert(4, y[:test_valid_index])

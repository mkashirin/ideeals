from itertools import chain
from typing import Any, Optional

try:
    import cupy as cb
except ModuleNotFoundError:
    import numpy as cb
from numpy.typing import NDArray

from ._typing import ConfusionMatrix, IndicesMap


def compute_mean_absolute_error(actual: NDArray, predicted: NDArray) -> float:
    """Compute mean absolute error metric for regression model predictions.

    Args:
        actual (NDArray): Actual target values.
        predicted (NDArray): Predicted target values.

    Returns:
        float: Mean absolute error of the model.

    Raises:
        ValueError: If actual and predicted arrays have non-broadcasting
            shapes.
    """
    error = float(cb.mean(cb.abs(actual - predicted)))
    return error


def compute_mean_squared_error(actual: NDArray, predicted: NDArray) -> float:
    """Compute root mean squared error metric for regression model predictions.

    Args:
        actual (NDArray): Actual target values.
        predicted (NDArray): Predicted target values.

    Returns:
        float: Mean squared error of the model.

    Raises:
        ValueError: If actual and predicted arrays have non-broadcasting
            shapes.
    """
    error = float(cb.mean(cb.power(actual - predicted, 2)))
    return error


def compute_root_mean_squared_error(
    actual: NDArray, predicted: NDArray
) -> float:
    """Compute root mean squared error metric for regression model predictions.

    Args:
        actual (NDArray): Actual target values.
        predicted (NDArray): Predicted target values.

    Returns:
        float: Root of mean squared error of the model.

    Raises:
        ValueError: If actual and predicted arrays have non-broadcasting
            shapes.
    """
    error = cb.sqrt(compute_mean_squared_error(actual, predicted))
    return error


def compute_accuracy(actual: NDArray, predicted: NDArray) -> float:
    """Compute accuracy for any model predictions.

    Args:
        actual (NDArray): Actual target values.
        predicted (NDArray): Predicted target values.

    Returns:
        float: Accuracy score of the model (% of correct predictions).

    Raises:
        ValueError: If actual and predicted arrays have non-broadcasting
            shapes.
    """
    accuracy = cb.sum(predicted == actual) / len(actual)
    return accuracy


def compute_confusion_matrix(
    actual: NDArray,
    predicted: NDArray,
    indices_map: Optional[IndicesMap] = None,
) -> ConfusionMatrix:
    """Compute confusion matrix and get it with indices map for classification
    model predictions.

    Args:
        actual (NDArray): Actual target values.
        predicted (NDArray): Predicted target values.
        indices_map (Optional[IndicesMap], optional): Dictionary, where keys
            are features names and values are integer indices in the confusion
            matrix. Defaults to None.

    Returns:
        ConfusionMatrix: Confusion matrix of the model with indicies map, which
        describes matrix alignment.

    Raises:
        ValueError: If actual and predicted arrays have non-broadcasting
            shapes.
    """

    def _map_to_integers(array, imap):
        for i, _ in enumerate(array):
            array[i] = imap[array[i]]
        return array

    actual_list, predicted_list = (
        list(chain.from_iterable(actual.tolist())),
        list(chain.from_iterable(predicted.tolist())),
    )
    concatenated = actual_list + predicted_list
    n_features = len(set(concatenated))
    if indices_map is None:
        indices_map = {
            key: val for key, val in zip(set(concatenated), range(n_features))
        }

    confusion_matrix = cb.zeros((n_features, n_features))
    mapped_actual, mapped_predicted = (
        _map_to_integers(actual_list, indices_map),
        _map_to_integers(predicted_list, indices_map),
    )
    for a, p in zip(mapped_actual, mapped_predicted):
        confusion_matrix[a, p] += 1
    confusion_matrix_with_map: ConfusionMatrix = confusion_matrix, indices_map

    return confusion_matrix_with_map


def compute_sensitivities_and_specificities(
    actual: NDArray, predicted: NDArray, as_array: bool = True
) -> Any:
    """Compute sensitivities and specificities for classification model
    predictions.

    Args:
        actual (NDArray): Actual target values.
        predicted (NDArray): Predicted target values.
        as_array (bool, optional): If True is passed, function will return
            regular NumPy NDArray, where rows correspond sensitivities to
            and specificities of features (those correspond to columns);
            otherwise function will return dict which describes upper-mentioned
            alignment explicitly. Defaults to True.

    Returns:
        Any: NumPy NDArray or dictionary of sensitivities and specificities
        (depends on ``as_array``).

    Raises:
        ValueError: If actual and predicted arrays have non-broadcasting
            shapes.
    """

    confusion_matrix, indices_map = compute_confusion_matrix(actual, predicted)
    n_features = len(indices_map)

    sensitivities, specificities = (list(), list())
    for i in range(n_features):
        true_positives = confusion_matrix[i, i]
        false_negatives = cb.sum(confusion_matrix[:, i])
        sensitivity = true_positives / (true_positives + false_negatives)
        sensitivities.append(sensitivity)

        upper_left = cb.sum(confusion_matrix[:i, :i])
        upper_right = cb.sum(confusion_matrix[:i, i + 1 :])
        lower_left = cb.sum(confusion_matrix[i + 1 :, :i])
        lower_right = cb.sum(confusion_matrix[i + 1 :, i + 1 :])
        true_negatives = cb.sum(
            (upper_left, upper_right, lower_left, lower_right)
        )
        false_positives = cb.sum(confusion_matrix[i])
        specificity = true_negatives / (true_negatives + false_positives)
        specificities.append(specificity)

    if not as_array:
        features_names = list(indices_map.keys())
        keys = ["sensitivities", "specificities"]
        metrics = sensitivities, specificities
        sensitivities_and_specificities = dict.fromkeys(keys)
        for outer_key, metric in zip(keys, metrics):
            sensitivities_and_specificities[outer_key] = {
                key: val for key, val in zip(features_names, metric)
            }
        return sensitivities_and_specificities

    sensitivities_and_specificities = cb.array([sensitivities, specificities])
    return sensitivities_and_specificities

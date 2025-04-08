from typing import Any, Dict, Literal, Tuple, Union

from numpy.typing import NDArray

# Abstract models
ArraysMap = Dict[str, NDArray]

# Evaluation metrics
IndicesMap = Dict[Any, int]
ConfusionMatrix = Tuple[NDArray, IndicesMap]
Selections = Union[
    Tuple[NDArray, NDArray, NDArray, NDArray],
    Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray],
]
Tweaks = Dict[str, Dict[str, Tuple[NDArray]]]

# Layers
WeightsOption = Literal["Glorot", "standard"]

# Manual models
SamplesBatch = Tuple[NDArray, NDArray]
WeightsMap = Dict[str, Union[NDArray, float]]
ComputationalMetadata = Dict[str, Union[Dict[str, NDArray], WeightsMap, float]]

# Optimizers
DecayType = Literal["exponential", "linear"]

# Preprocessing
StrategyOption = Literal["mean", "median", "constant"]

from typing import Protocol, Union

import numpy as np
import pandas as pd
import torch


class BuilderProtocol(Protocol):
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        ...


class TransformerProtocol(Protocol):
    def fit(self, X: np.ndarray) -> "TransformerProtocol":
        ...

    def fit_transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        ...

    def transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        ...

    def inverse_transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        ...

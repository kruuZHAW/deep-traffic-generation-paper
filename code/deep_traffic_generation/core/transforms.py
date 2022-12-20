from typing import List, Tuple, Union

import numpy as np
import torch


class PyTMinMaxScaler(object):
    """Pytorch implementation of sklearn.preprocessing.MinMaxScaler"""

    def __init__(
        self, feature_range=Tuple[float, float], *, copy=True, clip=False
    ) -> None:
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            # del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X: torch.Tensor) -> "PyTMinMaxScaler":
        """Compute the minimum and maximum to be used for later scaling.

        Args:
            X : tensor of shape (n_samples, n_features). The data used to
                compute the per-feature minimum and maximum used for later
                scaling along the features axis.

        Returns:
            self (object): Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X: torch.Tensor) -> "PyTMinMaxScaler":
        feature_range = self.feature_range

        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller"
                " than maximum. Got %s." % str(feature_range)
            )

        data_min, _ = torch.min(X, dim=0)
        data_max, _ = torch.max(X, dim=0)

        data_range = data_max - data_min
        data_range[data_range == 0.0] = 1.0
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - data_min * self.scale_

        self.data_min = data_min
        self.data_max = data_max
        self.data_range = data_range

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.partial_transform(X, idxs=np.arange(X.size(1)))

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.partial_inverse(X, idxs=np.arange(X.size(1)))

    def partial_transform(
        self, X: torch.Tensor, idxs: Union[int, List[int]]
    ) -> torch.Tensor:
        return X.mul(self.scale_[idxs]).add(self.min_[idxs])

    def partial_inverse(
        self, X: torch.Tensor, idxs: Union[int, List[int]]
    ) -> torch.Tensor:
        return X.sub(self.min_[idxs]).div(self.scale_[idxs])

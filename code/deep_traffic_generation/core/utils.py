# fmt: off
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import traj_dist.distance as tdist
# from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats._distn_infrastructure import rv_continuous
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from traffic.core import Traffic
from traffic.core.projection import EuroPP

from .protocols import BuilderProtocol

# from deep_traffic_generation.core.datasets import TrafficDataset


# fmt: on
def extract_features(
    traffic: Traffic,
    features: List[str],
    init_features: List[str] = [],
) -> np.ndarray:
    """Extract features from Traffic data according to the feature list.

    Parameters
    ----------
    traffic: Traffic
    features: List[str]
        Labels of the columns to extract from the underlying dataframe of
        Traffic object.
    init_features: List[str]
        Labels of the features to extract from the first row of each Flight
        underlying dataframe.
    Returns
    -------
    np.ndarray
        Feature vector `(N, HxL)` with `N` number of flights, `H` the number
        of features and `L` the sequence length.
    """
    X = np.stack(list(f.data[features].values.ravel() for f in traffic))

    if len(init_features) > 0:
        init_ = np.stack(
            list(f.data[init_features].iloc[0].values.ravel() for f in traffic)
        )
        X = np.concatenate((init_, X), axis=1)

    return X


def get_dataloaders(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    test_batch_size: Optional[int],
    num_workers: int = 5,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    val_size = int(train_size * val_ratio)
    train_size -= val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    if val_size > 0:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=test_batch_size
            if test_batch_size is not None
            else len(val_dataset),
            shuffle=True,
            num_workers=num_workers,
        )
    else:
        val_loader = None

    if test_size > 0:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size
            if test_batch_size is not None
            else len(val_dataset),
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


# fmt: off
def init_dataframe(
    data: np.ndarray, features: List[str], init_features: List[str] = [],
) -> pd.DataFrame:
    """ TODO:
    """
    # handle dense features (features)
    dense: np.ndarray = data[:, len(init_features):]
    nb_samples = data.shape[0]
    dense = dense.reshape(nb_samples, -1, len(features))
    nb_obs = dense.shape[1]
    # handle sparce features (init_features)
    if len(init_features) > 0:
        sparce = data[:, :len(init_features)]
        sparce = sparce[:, np.newaxis]
        sparce = np.insert(
            sparce, [1] * (nb_obs - 1), [np.nan] * len(init_features), axis=1
        )
        dense = np.concatenate((dense, sparce), axis=2)
        features = features + init_features

    # generate dataframe
    df = pd.DataFrame(
        {feature: dense[:, :, i].ravel() for i, feature in enumerate(features)}
    )
    return df


# fmt: on
def traffic_from_data(
    data: np.ndarray,
    features: List[str],
    init_features: List[str] = [],
    builder: Optional[BuilderProtocol] = None,
) -> Traffic:

    df = init_dataframe(data, features, init_features)

    if builder is not None:
        df = builder(df)

    return Traffic(df)


def plot_traffic(traffic: Traffic) -> Figure:
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(
            1, figsize=(5, 5), subplot_kw=dict(projection=EuroPP())
        )
        traffic[1].plot(ax, c="orange", label="reconstructed")
        traffic[0].plot(ax, c="purple", label="original")
        ax.legend()

    return fig


"""
    Function below from https://github.com/JulesBelveze/time-series-autoencoder
"""


def init_hidden(
    x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True
):
    """Initialize hidden.

    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(
            torch.zeros(num_dir, x.size(0), hidden_size)
        ).to(x.device)
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(x.device)


def build_weights(size: int, builder: rv_continuous, **kwargs) -> np.ndarray:
    """Build weight array according to a density law."""
    w = np.array(
        [builder.pdf(i / (size + 1), **kwargs) for i in range(1, size + 1)]
    )
    return w


def plot_clusters(traffic: Traffic, cluster_label: str = "cluster") -> Figure:
    assert (
        cluster_label in traffic.data.columns
    ), f"Underlying dataframe should have a {cluster_label} column"
    clusters = sorted(list(traffic.data[cluster_label].value_counts().keys()))
    n_clusters = len(clusters)
    # -- dealing with the grid
    if n_clusters > 3:
        nb_cols = 3
        nb_lines = n_clusters // nb_cols + ((n_clusters % nb_cols) > 0)

        with plt.style.context("traffic"):
            fig, axs = plt.subplots(
                nb_lines,
                nb_cols,
                figsize=(10, 15),
                subplot_kw=dict(projection=EuroPP()),
            )

            for n, cluster in enumerate(clusters):
                ax = axs[n // nb_cols][n % nb_cols]
                ax.set_title(f"cluster {cluster}")
                t_cluster = traffic.query(f"{cluster_label} == {cluster}")
                t_cluster.plot(ax, alpha=0.5)
                t_cluster.centroid(nb_samples=None, projection=EuroPP()).plot(
                    ax, color="red", alpha=1
                )
    else:
        with plt.style.context("traffic"):
            fig, axs = plt.subplots(
                n_clusters,
                figsize=(10, 15),
                subplot_kw=dict(projection=EuroPP()),
            )

            for n, cluster in enumerate(clusters):
                ax = axs[n]
                ax.set_title(f"cluster {cluster}")
                t_cluster = traffic.query(f"{cluster_label} == {cluster}")
                t_cluster.plot(ax, alpha=0.5)
                t_cluster.centroid(nb_samples=None, projection=EuroPP()).plot(
                    ax, color="red", alpha=1
                )
    return fig


def unpad_sequence(padded: torch.Tensor, lengths: torch.Tensor) -> List:
    return [padded[i][: lengths[i]] for i in range(len(padded))]


# def compare_xy(reconstruct: Traffic, ref: Traffic) -> pd.DataFrame:
#     res = {}
#     for f1 in tqdm(reconstruct):
#         f2 = ref[f1.flight_id]
#         aligned = f2.aligned_on_ils("LSZH").next()
#         if aligned is None:
#             continue
#         f2 = f2.before(aligned.start)
#         f1 = f1.before(f1.stop)
#         f2 = f2.before(f1.stop)
#         if f1 is None or f2 is None:
#             continue

#         X1, X2 = (
#             f1.resample(50).data[["x", "y"]].to_numpy(),
#             f2.resample(50).data[["x", "y"]].to_numpy(),
#         )

#         res[f1.flight_id] = dict(
#             dtw=tdist.dtw(X1, X2),
#             edr=tdist.edr(X1, X2),
#             erp=tdist.erp(X1, X2, g=np.zeros(2, dtype=float)),
#             frechet=tdist.frechet(X1, X2),
#             hausdorff=tdist.hausdorff(X1, X2),
#             lcss=tdist.lcss(X1, X2),
#             sspd=tdist.sspd(X1, X2),
#         )

#     return pd.DataFrame(res).T


def cumul_dist_plot(
    df: pd.DataFrame, scales: Dict[Any, Tuple[float, float]], domain: List[str]
):
    alt.data_transformers.disable_max_rows()

    base = alt.Chart(df)
    legend_config = dict(
        labelFontSize=12,
        titleFontSize=13,
        labelFont="Ubuntu",
        titleFont="Ubuntu",
        orient="none",
        legendY=430
        # offset=0,
    )

    chart = (
        alt.vconcat(
            *[
                base.transform_window(
                    cumulative_count="count()",
                    sort=[{"field": col}],
                    groupby=["generation", "reconstruction"],
                )
                .transform_joinaggregate(
                    total="count()", groupby=["generation", "reconstruction"]
                )
                .transform_calculate(
                    normalized=alt.datum.cumulative_count / alt.datum.total
                )
                .mark_line(clip=True)
                .encode(
                    alt.X(
                        col,
                        title="Distance",
                        scale=alt.Scale(domain=scales[col]),
                    ),
                    alt.Y("normalized:Q", title="Cumulative ratio"),
                    alt.Color(
                        "generation",
                        legend=alt.Legend(
                            title="Generation method", **legend_config
                        ),
                        scale=alt.Scale(domain=domain),
                    ),
                    alt.StrokeDash(
                        "reconstruction",
                        legend=alt.Legend(
                            title="Reconstruction method",
                            legendX=200,
                            **legend_config,
                        ),
                        scale=alt.Scale(
                            domain=["Navigational points", "Douglas-Peucker"]
                        ),
                    ),
                )
                .properties(title=col.upper(), height=150)
                for col in scales
            ]
        )
        .configure_view(stroke=None)
        .configure_title(font="Fira Sans", fontSize=16, anchor="start")
        .configure_axis(
            labelFont="Fira Sans",
            labelFontSize=14,
            titleFont="Ubuntu",
            titleFontSize=12,
        )
    )

    return chart

from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import pyproj
from cartopy import crs
from traffic.core.geodesy import destination

from .protocols import BuilderProtocol


class CollectionBuilder(BuilderProtocol):
    """Collection of builder instances."""

    def __init__(self, builders: List[BuilderProtocol]) -> None:
        self.builders = builders

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for builder in self.builders:
            data = builder(data)
        return data

    def __add__(self, other: "CollectionBuilder") -> "CollectionBuilder":
        return CollectionBuilder(builders=self.builders + other.builders)

    def __getitem__(self, idx: int) -> BuilderProtocol:
        return self.builders[idx]

    def append(self, builder: BuilderProtocol) -> None:
        self.builders.append(builder)


class IdentifierBuilder(BuilderProtocol):
    """Builder for flight identifiers (callsign, icao24, flight_id)"""

    def __init__(self, nb_samples: int, nb_obs: int) -> None:
        self.nb_samples = nb_samples
        self.nb_obs = nb_obs
        self.identifiers = np.array(
            [[f"TRAJ_{sample}"] * nb_obs for sample in range(nb_samples)]
        ).ravel()

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.assign(
            flight_id=self.identifiers,
            callsign=self.identifiers,
            icao24=self.identifiers,
        )


class LatLonBuilder(BuilderProtocol):
    """Builder for latitude and longitude data."""

    _available_foundations = ["xy", "azgs", "azgs_r"]

    def __init__(self, build_from: str, **kwargs):
        self.foundation = build_from
        self.builder = self.get_builder(build_from)
        self.kwargs = kwargs

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        lat, lon = self.builder(data)
        return data.assign(latitude=lat, longitude=lon)

    def get_builder(
        self, foundation: str
    ) -> Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
        if foundation == "xy":
            return self.from_xy
        elif foundation == "azgs":
            return self.from_azgs
        elif foundation == "azgs_r":
            return self.from_azgs_reverse
        else:
            raise ValueError(f"Unsupported foundation: {foundation}.")

    def from_xy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: comments"""
        projection: Union[pyproj.Proj, crs.Projection, None] = None

        # TODO: check data has x and y columns.

        if "projection" in self.kwargs.keys():
            projection = self.kwargs["projection"]

        if isinstance(projection, crs.Projection):
            projection = pyproj.Proj(projection.proj4_init)

        if projection is None:
            projection = pyproj.Proj(
                proj="lcc",
                ellps="WGS84",
                lat_1=data.y.min(),
                lat_2=data.y.max(),
                lat_0=data.y.mean(),
                lon_0=data.x.mean(),
            )

        transformer = pyproj.Transformer.from_proj(
            projection, pyproj.Proj("epsg:4326"), always_xy=True
        )
        lon, lat = transformer.transform(
            data.x.values,
            data.y.values,
        )

        return lat, lon

    def from_azgs(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: comments"""
        # TODO: check data

        lat = np.empty(len(data))
        lon = np.empty(len(data))

        for i in range(len(data)):
            if np.isnan(data.loc[i, "longitude"]) or np.isnan(
                data.loc[i, "latitude"]
            ):
                lon1 = lon[i - 1]
                lat1 = lat[i - 1]
                track = data.loc[i - 1, "track"]
                gs = data.loc[i - 1, "groundspeed"]
                delta_time = (
                    data.loc[i, "timestamp"] - data.loc[i - 1, "timestamp"]
                ).total_seconds()
                # TODO: find best coeff
                coeff = 0.99
                d = coeff * gs * delta_time * (1852 / 3600)
                lat2, lon2, _ = destination(lat1, lon1, track, d)
                lat[i] = lat2
                lon[i] = lon2
            else:
                lat[i] = data.loc[i, "latitude"]
                lon[i] = data.loc[i, "longitude"]

        return lat, lon

    def from_azgs_reverse(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        df = data.copy(deep=True)
        df["track"] = df["track"].values[::-1] - 180
        df["groundspeed"] = df["groundspeed"].values[::-1]
        df["timestamp"] = df["timestamp"].values[::-1]
        lat = np.empty(len(df))
        lon = np.empty(len(df))

        for i in range(len(df)):
            if np.isnan(df.loc[i, "longitude"]) or np.isnan(
                df.loc[i, "latitude"]
            ):
                lon1 = lon[i - 1]
                lat1 = lat[i - 1]
                track = df.loc[i - 1, "track"]
                gs = df.loc[i - 1, "groundspeed"]
                delta_time = (
                    df.loc[i - 1, "timestamp"] - df.loc[i, "timestamp"]
                ).total_seconds()
                # TODO: find best coeff
                coeff = 0.99
                d = coeff * gs * delta_time * (1852 / 3600)
                lat2, lon2, _ = destination(lat1, lon1, track, d)
                lat[i] = lat2
                lon[i] = lon2
            else:
                lat[i] = df.loc[i, "latitude"]
                lon[i] = df.loc[i, "longitude"]

        return lat[::-1], lon[::-1]


class TimestampBuilder(BuilderProtocol):
    """Builder for timestamp data."""

    def __init__(
        self, base_ts: pd.Timestamp = pd.Timestamp.today(tz="UTC")
    ) -> None:
        self.base_ts = base_ts.round(freq="S")

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        assert (
            "timedelta" in data.columns
        ), "timedelta column is missing from data."

        return data.assign(
            timestamp=pd.to_timedelta(data.timedelta, unit="s") + self.base_ts
        )

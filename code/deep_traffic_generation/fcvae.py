# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from deep_traffic_generation.core import FCN, VAE, cli_main
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDataset
from deep_traffic_generation.core.lsr import NormalLSR


# fmt: on
class FCVAE(VAE):
    """Fully-Connected Variational Autoencoder"""

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.example_input_array = torch.rand(
            (1, self.dataset_params["input_dim"])
        )

        # encoder
        self.encoder = FCN(
            input_dim=self.dataset_params["input_dim"],
            out_dim=self.hparams.h_dims[-1],
            h_dims=self.hparams.h_dims[:-1],
            h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )

        # Latent Space Regularization
        self.lsr = NormalLSR(
            input_dim=self.hparams.h_dims[-1],
            out_dim=self.hparams.encoding_dim,
        )

        # decoder
        self.decoder = FCN(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.dataset_params["input_dim"],
            h_dims=self.hparams.h_dims[::-1],
            h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )

        # self.out_activ = nn.Tanh()
        self.out_activ = nn.Identity()

    @classmethod
    def network_name(cls) -> str:
        return "fcvae"

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds FCVAE arguments to ArgumentParser.

        List of arguments:

            * ``--n_components``: Number of components for the Gaussian Mixture
              modelling the latent space. Default to :math:`1`.

        .. note::
            It adds also the argument of the inherited class `VAE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--n_components", dest="n_components", type=int, default=1
        )

        return parent_parser, parser


if __name__ == "__main__":
    cli_main(FCVAE, TrafficDataset, "linear", seed=42)

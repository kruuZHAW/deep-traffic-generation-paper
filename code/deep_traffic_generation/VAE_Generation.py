from deep_traffic_generation.tcvae import TCVAE
from deep_traffic_generation.fcvae import FCVAE
from deep_traffic_generation.core.datasets import TrafficDataset

import torch
import numpy as np

from os import walk
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, TypedDict


class SingleStageVAE:
    def __init__(
        self,
        X: TrafficDataset,  # Traffic dataset used to train the first VAE stage
        sim_type: str = "generation",
    ):
        super().__init__()

        self.X = X
        self.sim_type = sim_type

        if sim_type not in ["generation", "reconstruction"]:
            raise ValueError(
                "Invalid sim type. Expected one of: %s"
                % ["generation", "reconstruction"]
            )

    def load(
        self,
        path: str,
        dataset_params: TypedDict,
    ):
        filenames = next(walk(path + "/checkpoints/"), (None, None, []))[2]

        if path.split("/")[-2] == "tcvae":
            self.type = "TCVAE"
            self.VAE = TCVAE.load_from_checkpoint(
                path + "/checkpoints/" + filenames[0],
                # hparams_file=path + "/hparams.yaml",
                dataset_params=dataset_params,
            )

        elif path.split("/")[-2] == "fcvae":
            self.type = "FCVAE"
            self.VAE = FCVAE.load_from_checkpoint(
                path + "/checkpoints/" + filenames[0],
                # hparams_file=path + "/hparams.yaml",
                dataset_params=dataset_params,
            )

        self.VAE.eval()

    def latent_space(
        self, n_samples: int
    ):  # Gives the latent spaces of the VAE, and n_sample generated points with the prior

        h = self.VAE.encoder(self.X.data)
        q = self.VAE.lsr(h)
        z_1 = q.rsample()

        p_z = self.VAE.lsr.get_prior()
        z_gen = p_z.sample(torch.Size([n_samples])).squeeze(1)

        z_embeddings = np.concatenate(
            (z_1.detach().numpy(), z_gen.detach().numpy()), axis=0
        )

        return z_embeddings

    def decode(self, latent):  # decode some given latent variables

        if self.type == "TCVAE":
            reco_x = self.VAE.decoder(latent.to(self.VAE.device)).cpu()
            # make sure the first timedelta predicted is 0
            # reco_x[:, self.VAE.hparams.features.index("timedelta")] = 0
            decoded = reco_x.detach().transpose(1, 2)
            decoded = decoded.reshape((decoded.shape[0], -1))
            decoded = self.X.scaler.inverse_transform(decoded)

        if self.type == "FCVAE":
            reco_x = self.VAE.decoder(latent.to(self.VAE.device)).cpu()
            # make sure the first timedelta predicted is 0
            # reco_x[:, self.VAE.hparams.features.index("timedelta")] = 0
            decoded = self.X.scaler.inverse_transform(reco_x.detach().numpy())

        return decoded

    def fit(self, X, **kwargs):
        return self

    def sample(self, n_samples: int):  # Tuple[ndarray[float], ndarray[float]]

        with torch.no_grad():

            # Even for generation, need to run lsr to compute prior_means for VampPrior
            h = self.VAE.encoder(self.X.data[:n_samples])
            q = self.VAE.lsr(h)

            if self.sim_type == "generation":
                p_z = self.VAE.lsr.get_prior()
                z = p_z.sample(torch.Size([n_samples])).squeeze(1)

            if self.sim_type == "reconstruction":
                z = q.rsample()

            gen_x = self.VAE.decoder(z.to(self.VAE.device)).cpu()
            # make sure the first timedelta predicted is 0
            # gen_x[:, self.VAE.hparams.features.index("timedelta")] = 0

        if self.type == "TCVAE":
            gen_x = gen_x.detach().transpose(1, 2).reshape(gen_x.shape[0], -1)

        return gen_x, 0

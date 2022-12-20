"""Script that generate synthetic trajectories assuming that the VAE has been trained previously trained. Pseudo-inputs only work for VampPrior TCVAE.

Arguments:
    - training_name (str): Name of traffic object used to train the VAE
    - fcvae_version (str): lightning log version of fcvae (e.g. version_0)
    - tcvae_version (str): lightning log version of tcvae (should imperatively trained with VampPrior)

Outputs:
    - Traffic object of reconstruction for fcvae
    - Traffic object of reconstruction for tcvae
    - Clustered latent space for fcvae
    - List of traffic objects for reconstructed trajectories for fcvae corresponding to the clustering
    - Clustered latent space for tcvae
    - List of traffic objects for reconstructed trajectories for fcvae corresponding to the clustering
    - Latent space of generated trajectories for TCVAE
    - Traffic object of generated trajectories for TCVAE (flow 1)
    - Traffic object of generated trajectories for TCVAE (flow 2)

"""

import click
from typing import List

import torch
import numpy as np
import pandas as pd
from traffic.algorithms.generation import Generation
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from traffic.core import Traffic
from deep_traffic_generation.VAE_Generation import SingleStageVAE
from deep_traffic_generation.core.datasets import TrafficDataset

def loading(training_name: str, fcvae_version: str, tcvae_version: str):

    #features and scaler are hardcoded: Only those are used in the paper
    dataset_fcvae = TrafficDataset.from_file(
        "../../data/" + training_name,
        features=["track", "groundspeed", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        shape= "linear",
        info_params={"features": ["latitude", "longitude"], "index": -1},
    )


    dataset_tcvae = TrafficDataset.from_file(
        "../../data/" + training_name,
        features=["track", "groundspeed", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        shape= "image",
        info_params={"features": ["latitude", "longitude"], "index": -1},
    )

    path_fcvae = "/".join(["deep_traffic_generation/lightning_logs/fcvae", vae_version])
    path_tcvae = "/".join(["deep_traffic_generation/lightning_logs/tcvae", vae_version])

    t_fcvae = SingleStageVAE(X=dataset_fcvae, sim_type="generation")
    t_fcvae.load(path_fcvae, dataset_fcvae.parameters)
    g_tcvae = Generation(
        generation=t_fcvae,
        features=t_fcvae.VAE.hparams.features,
        scaler=dataset_fcvae.scaler,
    )

    t_tcvae = SingleStageVAE(X=dataset_tcvae, sim_type="generation")
    t_tcvae.load(path_tcvae, dataset_tcvae.parameters)
    g_tcvae = Generation(
        generation=t_tcvae,
        features=t_tcvae.VAE.hparams.features,
        scaler=dataset_tcvae.scaler,
    )

    return dataset_fcvae, dataset_tcvae, t_fcvae, t_tcvae, g_fcvae, g_tcvae

def reconstruction(dataset_fcvae: TrafficDataset, 
    dataset_tcvae: TrafficDataset, 
    t_fcvae: SingleStageVAE, 
    t_tcvae: SingleStageVAE, 
    g_fcvae: Generation, 
    g_tcvae: Generation,
    traffic_name: str, 
    j: int = 10795):
    
    traffic = Traffic.from_file("../../data/" + training_name)

    click.echo("Reconstruction for FCVAE...")
    original_fcvae = dataset_fcvae.data[j].unsqueeze(0)
    if len(original_fcvae.shape) >= 3:
        original_fcvae = original_fcvae.transpose(1, 2).reshape((original_fcvae.shape[0], -1))
    original_fcvae = dataset_fcvae.scaler.inverse_transform(original_fcvae)
    original_traf_fcvae = g_fcvae.build_traffic(
        original_fcvae,
        coordinates=dict(latitude=47.546585, longitude=8.447731),
        forward=False,
    )
    reconstructed_fcvae = t_fcvae.decode(
        t_fcvae.VAE.encoder(dataset_fcvae.data[j].unsqueeze(0))
    )
    reconstructed_traf_fcvae = g_fcvae.build_traffic(
        reconstructed_fcvae,
        coordinates=dict(latitude=47.546585, longitude=8.447731),
        forward=False,
    )
    reconstruction_traf_fcvae = traffic[j] + reconstructed_traf_fcvae

    click.echo("Reconstruction for TCVAE...")
    original_tcvae = dataset_tcvae.data[j].unsqueeze(0)
    if len(original_tcvae.shape) >= 3:
        original_tcvae = original_tcvae.transpose(1, 2).reshape((original_tcvae.shape[0], -1))
    original_tcvae = dataset_tcvae.scaler.inverse_transform(original_tcvae)
    original_traf_tcvae = g_tcvae.build_traffic(
        original_tcvae,
        coordinates=dict(latitude=47.546585, longitude=8.447731),
        forward=False,
    )
    reconstructed_tcvae = t_tcvae.decode(
        t_tcvae.VAE.encoder(dataset_tcvae.data[j].unsqueeze(0))
    )
    reconstructed_traf_tcvae = g_tcvae.build_traffic(
        reconstructed_tcvae,
        coordinates=dict(latitude=47.546585, longitude=8.447731),
        forward=False,
    )
    reconstruction_traf_tcvae = traffic[j] + reconstructed_traf_tcvae

    return reconstruction_traf_fcvae, reconstruction_traf_tcvae

def clustering(dataset_fcvae: TrafficDataset, 
    dataset_tcvae: TrafficDataset, 
    t_fcvae: SingleStageVAE, 
    t_tcvae: SingleStageVAE, 
    g_fcvae: Generation, 
    g_tcvae: Generation): 

    click.echo("Clustering for FCVAE...")
    Z_fcvae = t_fcvae.latent_space(1)
    pca_fcvae = PCA(n_components=2).fit(Z_fcvae)
    Z_embedded_fcvae = pca_fcvae.transform(Z_fcvae)
    labels_fcvae = GaussianMixture(n_components=4, random_state=0).fit_predict(Z_embedded_fcvae)
    Z_embedded_fcvae = np.append(Z_embedded_fcvae, np.expand_dims(labels_fcvae, axis=1), axis=1)
    Z_embedded_fcvae = pd.DataFrame(Z_embedded_fcvae, columns=["X1", "X2", "label"])

    traffics_fcvae = []
    for i in np.unique(labels_fcvae):
        print("fcvae traffic : ", i)
        decoded = t_fcvae.decode(torch.Tensor(Z[labels_fcvae == i]))
        traf_clust = g_FCVAE.build_traffic(
            decoded,
            coordinates=dict(latitude=47.546585, longitude=8.447731),
            forward=False,
        )
        traf_clust = traf_clust.assign(cluster=lambda x: i)
        traffics_fcvae.append(traf_clust)

    click.echo("Clustering for TCVAE...")
    Z_tcvae = t_tcvae.latent_space(1)
    pca_tcvae = PCA(n_components=2).fit(Z_tcvae)
    Z_embedded_tcvae = pca_tcvae.transform(Z)
    labels_tcvae = GaussianMixture(n_components=7, random_state=0).fit_predict(Z_embedded_tcvae)
    Z_embedded_tcvae = np.append(Z_embedded_tcvae, np.expand_dims(labels_tcvae, axis=1), axis=1)
    Z_embedded_tcvae = pd.DataFrame(Z_embedded_tcvae, columns=["X1", "X2", "label"])

    traffics_tcvae = []
    for i in np.unique(labels_tcvae):
        print("tcvae traffic : ", i)
        decoded = t_tcvae.decode(torch.Tensor(Z[labels_tcvae == i]))
        traf_clust = g_tcvae.build_traffic(
            decoded,
            coordinates=dict(latitude=47.546585, longitude=8.447731),
            forward=False,
        )
        traf_clust = traf_clust.assign(cluster=lambda x: i)
        traffics_tcvae.append(traf_clust)


    return Z_embedded_fcvae, Z_embedded_tcvae, traffics_fcvae, traffics_tcvae

#Generate in within particular VampPrior componenents for TCVAE
def generate_vamp_tcvae(training_data: TrafficDataset, t: SingleStageVAE, g: Generation, indexes: List[int] = [39, 91], n_gen: int = 100):

    #Calculating Pseudo-inputs
    pseudo_X = t.VAE.lsr.pseudo_inputs_NN(t.VAE.lsr.idle_input)
    pseudo_X = pseudo_X.view((pseudo_X.shape[0], training_data.data.shape[1], training_data.data.shape[2]))
    pseudo_h = t.VAE.encoder(pseudo_X)
    pseudo_means = t.VAE.lsr.z_loc(pseudo_h)
    pseudo_scales = (t.VAE.lsr.z_log_var(pseudo_h) / 2).exp()

    #Generating in 2 specific VampPrior components
    dist1 = torch.distributions.Independent(
        torch.distributions.Normal(pseudo_means[indexes[0]], pseudo_scales[indexes[0]]), 1
    )
    gen1 = dist1.sample(torch.Size([n_gen]))

    dist2 = torch.distributions.Independent(
        torch.distributions.Normal(pseudo_means[indexes[1]], pseudo_scales[indexes[1]]), 1
    )
    gen2 = dist2.sample(torch.Size([n_gen]))

    #Decoding back into trajectories: first is pseudo-input
    decode1 = t.decode(
        torch.cat((pseudo_means[indexes[0]].unsqueeze(0), gen1), axis=0)
    )
    decode2 = t.decode(
        torch.cat((pseudo_means[indexes[1]].unsqueeze(0), gen2), axis=0)
    )

    #Neural net don't predict exaclty timedelta = 0 for the first observation
    decode1[:, 3] = 0
    decode2[:, 3] = 0

    #Building Traffic object
    traf_gen1 = g.build_traffic(
        decode1,
        coordinates=dict(latitude=47.546585, longitude=8.447731), #Lat/lon hardcoded because that's the ending points of the training dataset
        forward=False,
    )
    traf_gen1 = traf_gen1.assign(gen_number=lambda x: 1)

    traf_gen2 = g.build_traffic(
        decode2,
        coordinates=dict(latitude=47.546585, longitude=8.447731),
        forward=False,
    )
    traf_gen2 = traf_gen2.assign(gen_number=lambda x: 2)

    #Projecting Latent Space in 2 dimensions
    z_train = t.latent_space(1)
    gen = torch.cat((gen1, gen2, pseudo_means[indexes]), axis=0)
    concat = np.concatenate((z_train, gen.detach().numpy()))
    pca = PCA(n_components=2).fit(concat[: -len(gen)])
    gen_embedded = pca.transform(concat)

    gen_embedded = pd.DataFrame(gen_embedded, columns=["X1", "X2"])
    gen_embedded["type"] = np.nan
    gen_embedded.type[-(2 * n_gen + 2) :] = "GEN1"
    gen_embedded.type[-(n_gen + 2) :] = "GEN2"
    gen_embedded.type[-2:] = "PI1"
    gen_embedded.type[-1:] = "PI2"

    return gen_embedded, traf_gen1, traf_gen2


#####
@click.command()
@click.argument("training_name", type=str)
@click.argument("fcvae_version", type=str)
@click.argument("tcvae_version", type=str)


def main(
    training_name:  str,
    fcvae_version:  str,
    tcvae_version:  str,
):

    click.echo("Loading VAEs...")
    dataset_fcvae, dataset_tcvae, t_fcvae, t_tcvae, g_fcvae, g_tcvae = loading(training_name, fcvae_version, tcvae_version)

    click.echo("Building Reconstruction...")
    reconstruction_traf_fcvae, reconstruction_traf_tcvae = reconstruction(dataset_fcvae, dataset_tcvae, t_fcvae, t_tcvae, g_fcvae, g_tcvae, traffic_name)
    reconstruction_traf_fcae.to_pickle("../../results/reconstruction/reconstruction_fcae.pkl")
    reconstruction_traf_tcae.to_pickle("../../results/reconstruction/reconstruction_tcae.pkl")

    click.echo("Building Clusterings...")
    Z_embedded_fcvae, Z_embedded_tcvae, traffics_fcvae, traffics_tcvae = clustering(dataset_fcvae, dataset_tcvae, t_fcvae, t_tcvae, g_fcvae,g_tcvae)
    Z_embedded_fcvae.to_pickle("../../results/clustering/Z_embedded_fcvae.pkl")
    with open("../../results/clustering/traffics_clust_fcvae.pkl", "wb") as f:
        pickle.dump(traffics_fcvae, f)
    Z_embedded_tcvae.to_pickle("../../results/clustering/Z_embedded_tcvae.pkl")
    with open("../../results/clustering/traffics_clust_tcvae.pkl", "wb") as f:
        pickle.dump(traffics_tcvae, f)

    click.echo("Generation VampPrior...")
    gen_embedded, traf_gen1, traf_gen2 = generate_vamp_tcvae(dataset_tcvae, t_tcvae, g_tcvae, indexes = [262, 787], n_gen = 100)
        gen_embedded.to_pickle("../../results/generation/latent_space_vampprior_tcvae.pkl")
        traf_gen1.to_pickle("../../results/generation/tcvae_traf_gen1.pkl")
        traf_gen2.to_pickle("../../results/generation/tcvae_traf_gen2.pkl")

if __name__ == "__main__":
    main()

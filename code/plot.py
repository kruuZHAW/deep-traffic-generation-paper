"""Script that plots dataset generated in generation.py

Arguments:

Outputs:

"""

import click
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import altair as alt

from traffic.core import Traffic
from traffic.core.projection import EuroPP, PlateCarree
from traffic.data import airports
from traffic.data import navaids

def plot_generation_tcvae(latent_space_path: str, traf_gen1_path: str, traf_gen2_path: str):

    Z_gen = pd.read_pickle(latent_space_path)
    traf_gen1 = Traffic.from_file(traf_gen1_path)
    traf_gen2 = Traffic.from_file(traf_gen2_path)

    #Plot TCVAE generation
    with plt.style.context("traffic"):
        fig = plt.figure(figsize=(17, 12))
        ax0 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222, projection=EuroPP())

        ax0.scatter(
            Z_gen.query("type.isnull()").X1,
            Z_gen.query("type.isnull()").X2,
            c="#bab0ac",
            s=4,
            label="Observed",
        )
        ax0.scatter(
            Z_gen.query("type == 'GEN1'").X1,
            Z_gen.query("type == 'GEN1'").X2,
            c="#9ecae9",
            s=8,
            label="Generation pseudo_input 1",
        )
        ax0.scatter(
            Z_gen.query("type == 'GEN2'").X1,
            Z_gen.query("type == 'GEN2'").X2,
            c="#ffbf79",
            s=8,
            label="Generation pseudo-input 2",
        )
        ax0.scatter(
            Z_gen.query("type == 'PI1'").X1,
            Z_gen.query("type == 'PI1'").X2,
            c="#4c78a8",
            s=50,
            label="Pseudo-input 1",
        )
        ax0.scatter(
            Z_gen.query("type == 'PI2'").X1,
            Z_gen.query("type == 'PI2'").X2,
            c="#f58518",
            s=50,
            label="Pseudo-input 2",
        )
        ax0.set_title("Latent Space", fontsize=18)

        legend = ax0.legend(loc="upper left", fontsize=12)
        legend.get_frame().set_edgecolor("none")
        legend.legendHandles[0]._sizes = [50]
        legend.legendHandles[1]._sizes = [50]
        legend.legendHandles[2]._sizes = [50]

        ax1.set_title("Generated synthetic trajectories", pad=0, fontsize=18)

        traf_gen1.plot(ax1, alpha=0.2, color="#9ecae9")
        traf_gen1["TRAJ_0"].plot(ax1, color="#4c78a8", lw=2)
        traf_gen1["TRAJ_0"].at_ratio(0.5).plot(
            ax1,
            color="#4c78a8",
            zorder=5,
            text_kw={"s": None},
        )

        traf_gen2.plot(ax1, alpha=0.2, color="#ffbf79")
        traf_gen2["TRAJ_0"].plot(ax1, color="#f58518", lw=2)
        traf_gen2["TRAJ_0"].at_ratio(0.5).plot(
            ax1,
            color="#f58518",
            zorder=5,
            text_kw={"s": None},
        )

        airports["LSZH"].point.plot(ax1)
        fig.tight_layout()
        plt.subplots_adjust(
            left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0
        )

    fig.savefig("../results/figures/figure_12.png", transparent=False, dpi=300)

    #Plot altitude + ground speed flows of traf_gen1 and traf_gen2
    # Just put the pseudo-input at the end for display
    copy_traf_1 = traf_gen1
    a = copy_traf_1["TRAJ_0"].assign(flight_id="TRAJ_999")
    copy_traf_1 = copy_traf_1 + a

    copy_traf_2 = traf_gen2
    b = copy_traf_2["TRAJ_0"].assign(flight_id="TRAJ_999")
    copy_traf_2 = copy_traf_2 + b

    chart1 = alt.layer(
        *(
            flight.chart().encode(
                x=alt.X(
                    "timedelta",
                    title="timedelta (in s)",
                ),
                y=alt.Y("altitude", title=None),
                opacity=alt.condition(
                    alt.datum.flight_id == "TRAJ_999",
                    alt.value(1),
                    alt.value(0.2),
                ),
                color=alt.condition(
                    alt.datum.flight_id == "TRAJ_999",
                    alt.value("#4c78a8"),
                    alt.value("#9ecae9"),
                ),
            )
            for flight in copy_traf_1
        )
    ).properties(title="altitude (in ft)")

    chart2 = alt.layer(
        *(
            flight.chart().encode(
                x=alt.X(
                    "timedelta",
                    title="timedelta (in s)",
                ),
                y=alt.Y("groundspeed", title=None),
                opacity=alt.condition(
                    alt.datum.flight_id == "TRAJ_999",
                    alt.value(1),
                    alt.value(0.2),
                ),
                color=alt.condition(
                    alt.datum.flight_id == "TRAJ_999",
                    alt.value("#4c78a8"),
                    alt.value("#9ecae9"),
                ),
            )
            for flight in copy_traf_1
        )
    ).properties(title="groundspeed (in kts)")

    chart3 = alt.layer(
        *(
            flight.chart().encode(
                x=alt.X(
                    "timedelta",
                    title="timedelta (in s)",
                ),
                y=alt.Y("altitude", title=None),
                opacity=alt.condition(
                    alt.datum.flight_id == "TRAJ_999",
                    alt.value(1),
                    alt.value(0.2),
                ),
                color=alt.condition(
                    alt.datum.flight_id == "TRAJ_999",
                    alt.value("#f58518"),
                    alt.value("#ffbf79"),
                ),
            )
            for flight in copy_traf_2
        )
    ).properties(title="altitude (in ft)")

    chart4 = alt.layer(
        *(
            flight.chart().encode(
                x=alt.X(
                    "timedelta",
                    title="timedelta (in s)",
                ),
                y=alt.Y("groundspeed", title=None),
                opacity=alt.condition(
                    alt.datum.flight_id == "TRAJ_999",
                    alt.value(1),
                    alt.value(0.2),
                ),
                color=alt.condition(
                    alt.datum.flight_id == "TRAJ_999",
                    alt.value("#f58518"),
                    alt.value("#ffbf79"),
                ),
            )
            for flight in copy_traf_2
        )
    ).properties(title="groundspeed (in kts)")

    plots = (
        alt.vconcat(alt.hconcat(chart1, chart2), alt.hconcat(chart3, chart4))
        .configure_title(fontSize=18)
        .configure_axis(labelFontSize=12, titleFontSize=14)
    )

    plots.save('../results/figures/figure_13.html', scale_factor=2.0)
    
def plot_reconstruction(reconstruction_fcvae_path: str, reconstruction_tcvae_path: str):
    
    reconstruction_fcvae = Traffic.from_file(reconstruction_fcvae_path)
    reconstruction_tcvae = Traffic.from_file(reconstruction_tcvae_path)
    
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(
            1, 2, figsize=(13, 8), subplot_kw=dict(projection=EuroPP())
        )

        ax[0].set_title("FCVAE reconstruction", pad=20, fontsize=20)
        reconstruction_fcvae[0].plot(ax[0], lw=2)
        reconstruction_fcvae[1].plot(ax[0], lw=2)

        ax[1].set_title("TCVAE reconstruction", pad=20, fontsize=20)
        reconstruction_tcvae[0].plot(ax[1], lw=2, label="original")
        reconstruction_tcvae[1].plot(ax[1], lw=2, label="reconstructed")
        ax[1].set_extent(ax[0].get_extent(crs=PlateCarree()))
        legend = fig.legend(
            loc="lower center", bbox_to_anchor=(0.5, 0.2), ncol=2, fontsize=18
        )
        legend.get_frame().set_edgecolor("none")

    fig.savefig("../results/figures/figure_6.png", transparent=False, dpi=300)
    
def plot_clustering(Z_fcvae_path: str, Z_tcvae_path: str, traffics_fcvae_path: str, traffics_tcvae_path: str):
    with open(traffics_fcvae_path, "rb") as f:
        traffics_fcvae = mynewlist = pickle.load(f)

    with open(traffics_tcvae_path, "rb") as f:
        traffics_tcvae = mynewlist = pickle.load(f)

    Z_fcvae = pd.read_pickle(Z_fcvae_path)
    Z_tcvae = pd.read_pickle(Z_tcvae_path)
    
    color_cycle = "#a6cee3 #1f78b4 #b2df8a #33a02c #fb9a99 #e31a1c #fdbf6f #ff7f00 #cab2d6 #6a3d9a #ffff99 #b15928".split()
    
    colors = [color_cycle[int(i)] for i in Z_fcvae.label]
    with plt.style.context("traffic"):
        fig = plt.figure(figsize=(30, 15))
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection=EuroPP())
        

        ax0.scatter(Z_fcvae.X1, Z_fcvae.X2, s=4, c = colors)
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_title("FCVAE latent space", fontsize=30, pad=18)
        ax0.grid(False)

        ax1.figure
        ax1.set_title("FCVAE reconstructed trajectories", fontsize=30, pad=18)
        for i, traf in enumerate(traffics_fcvae) :
            traf.plot(ax1, alpha=0.2, color = color_cycle[i])
    fig.savefig("../results/figures/figure_7.png", transparent=False, dpi=300)
    
    colors = [color_cycle[int(i)] for i in Z_tcvae.label]
    with plt.style.context("traffic"):
        fig = plt.figure(figsize=(30, 15))
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection=EuroPP())
        

        ax0.scatter(Z_tcvae.X1, Z_tcvae.X2, s=4, c = colors)
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_title("TCVAE latent space", fontsize=30, pad=18)
        ax0.grid(False)

        ax1.figure
        ax1.set_title("TCVAE reconstructed trajectories", fontsize=30, pad=18)
        for i, traf in enumerate(traffics_tcvae) :
            traf.plot(ax1, alpha=0.2, color = color_cycle[i])
    fig.savefig("../results/figures/figure_8.png", transparent=False, dpi=300)
    
    
    
def main(
):

    click.echo("Plotting generation TCVAE...")
    plot_generation_tcvae("../results/generation/latent_space_vampprior_tcvae.pkl", 
                          "../results/generation/tcvae_traf_gen1.pkl",
                          "../results/generation/tcvae_traf_gen2.pkl")
    
    click.echo("Plotting reconstruction...")
    plot_reconstruction("../results/reconstruction/reconstruction_fcvae.pkl",
                        "../results/reconstruction/reconstruction_tcvae.pkl")
    
    click.echo("Plotting Clustering...")
    plot_clustering("../results/clustering/Z_embedded_fcvae.pkl",
                    "../results/clustering/Z_embedded_tcvae.pkl",
                    "../results/clustering/traffics_clust_fcvae.pkl",
                    "../results/clustering/traffics_clust_tcvae.pkl")


if __name__ == "__main__":
    main()
    
    
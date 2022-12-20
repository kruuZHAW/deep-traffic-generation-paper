"""Script that plots dataset generated in generation.py

Arguments:

Outputs:

"""

import glob
import click
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

from traffic.core import Traffic
from traffic.core.projection import EuroPP
from traffic.data import airports
from traffic.data import navaids

#Loading Data
click.echo("Loading synthetic traffics...")
synth_traf_path = glob.glob("../../results/synth_traf_*")[0]
synth_traf = Traffic.from_file(synth_traf_path)

if len(glob.glob("../../results/latent_space_*")) > 0:
    latent_sapce_path = glob.glob("../../results/latent_space_*")[0]
    taf_gen1_path = glob.glob("../../results/*traf_gen1*")[0]
    taf_gen2_path = glob.glob("../../results/*traf_gen2*")[0]

    Z_gen = pd.read_pickle(latent_sapce_path)
    traf_gen1 = Traffic.from_file(taf_gen1_path)
    traf_gen2 = Traffic.from_file(taf_gen2_path)

#Plot traffic of random trajectories synth_traf
click.echo("Plotting random synthetic trajectories...")
with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        1, 1, figsize=(15, 15), subplot_kw=dict(projection=EuroPP()), dpi=300
    )
    synth_traf.plot(ax, alpha=0.2)

    k = 0
    synth_traf[k].plot(ax, color="#1f77b4", lw=1.5)
    synth_traf[k].at_ratio(0.8).plot(
        ax,
        color="#1f77b4",
        zorder=3,
        s=600,
        shift=dict(units="dots", x=-60, y=60),
        text_kw=dict(
            fontname="Fira Sans",
            # fontSize=18,
            ha="right",
            bbox=dict(
                boxstyle="round",
                edgecolor="none",
                facecolor="white",
                alpha=0.7,
                zorder=5,
            ),
        ),
    )

    airports["LSZH"].plot(ax, footprint=False, runways=dict(lw=1), labels=False)

    navaids["OSNEM"].plot(
        ax,
        zorder=5,
        marker="^",
        shift=dict(units="dots", x=45, y=-45),
        text_kw={"s": "FAP", 
            # "fontSize": 18, 
            "va": "center"},
    )

    fig.savefig("../../results/synthetic_trajectories.png", transparent=False, dpi=300)

if len(glob.glob("../../results/latent_space_*")) > 0:
    #Plot latent space + traf_gen1 + traf_gen2
    click.echo("Plotting synthetic traffic flows...")
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

    fig.savefig("../../results/generation_flows.png", transparent=False, dpi=300)

    #Plot altitude + ground speed flows of traf_gen1 and traf_gen2
    click.echo("Plotting atltitude and groundspeed...")
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

    plots.save('alt_gs_gen.html', scale_factor=2.0)
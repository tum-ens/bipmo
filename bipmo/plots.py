"""
Plots module for the bipmo biogas plant model defined in biogas_plant_models.py module
"""

import matplotlib.pyplot as plt
import matplotlib.dates
import pandas as pd
import numpy as np
import os

import fledge.data_interface
import fledge.der_models


# Plot results.
def generate_biogas_plant_plots(
        results: fledge.data_interface.ResultsDict,
        bg: fledge.der_models.FlexibleBiogasPlantModel,
        results_path: str,
        price_timeseries: pd.DataFrame = None,
        in_per_unit: bool = True,
):

    der_name = bg.der_name
    # Plot settings
    figsize = [7.8, 2.6]
    linewidth = 1.5
    legend_font_size = 12

    colors = list(color['color'] for color in matplotlib.rcParams['axes.prop_cycle'])
    show_grid = True
    if len(bg.timesteps) > 25:
        x_label_date_format = '%m/%d'
        x_axis_label = 'Date'
    else:
        x_label_date_format = '%H:%M'
        x_axis_label = 'Time'

    for output in bg.outputs:
        # Create plot.
        if in_per_unit and (not any(np.isinf(bg.output_maximum_timeseries[output].values))):
            maximum = bg.output_maximum_timeseries[output] / bg.output_maximum_timeseries[output]
            minimum = bg.output_minimum_timeseries[output] / bg.output_maximum_timeseries[output]
            optimum = results['output_vector'][output] / bg.output_maximum_timeseries[output]
        else:
            maximum = bg.output_maximum_timeseries[output]
            minimum = bg.output_minimum_timeseries[output]
            optimum = results['output_vector'][output]

        if ((bg.scenario_name + '_storage_content_m3') in output) or \
                ((bg.scenario_name + '_prod_biogas_m3_s-1') in output):
            drawstlye = 'default'
        else:
            drawstlye = 'steps-post'

        fig, (ax1, lax) = plt.subplots(ncols=2, figsize=figsize, gridspec_kw={"width_ratios": [100, 1]})
        ax1.set_title(f"DER {der_name} – {output}")
        ax1.plot(
            bg.timesteps,
            maximum,
            label='Maximum',
            drawstyle='steps-post',
            color=colors[2],
            linewidth=linewidth)
        ax1.plot(
            minimum,
            label='Minimum',
            drawstyle='steps-post',
            color=colors[1],
            linewidth=linewidth)
        ax1.plot(
            optimum,
            label='Optimal',
            drawstyle=drawstlye,
            color=colors[0],
            linewidth=linewidth)
        ax1.grid(show_grid)
        ax1.set_xlabel(x_axis_label)
        ax1.set_ylabel(f'{output}')

        # ax2 = plt.twinx(ax1)
        ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(x_label_date_format))
        ax1.set_xlim((bg.timesteps[0].toordinal(), bg.timesteps[-1].toordinal()))
        ax1.set_xlabel(x_axis_label)
        # ax2.set_ylabel('Power [p.u.]') if in_per_unit else ax2.set_ylabel('Power [W]')
        # ax2.set_ylim((0.0, 1.0)) if in_per_unit else ax2.set_ylim((0.0, 30.0))
        h1, l1 = ax1.get_legend_handles_labels()
        # h2, l2 = ax2.get_legend_handles_labels()
        lax.legend((h1), (l1), borderaxespad=0, prop={'size': legend_font_size})
        lax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'{der_name}_{output}.png'))
        plt.show()
        plt.close()

    for control in bg.controls:
        # Create plot.
        fig, (ax1, lax) = plt.subplots(ncols=2, figsize=figsize, gridspec_kw={"width_ratios": [100, 1]})
        ax1.set_title(f'DER {der_name}')
        ax1.plot(
            results['control_vector'][control],
            label='Optimal',
            drawstyle='steps-post',
            color=colors[0],
            linewidth=linewidth)
        ax1.grid(show_grid)
        ax1.set_xlabel(x_axis_label)
        ax1.set_ylabel(f'{control}')

        # ax2 = plt.twinx(ax1)
        ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(x_label_date_format))
        ax1.set_xlim((bg.timesteps[0].toordinal(), bg.timesteps[-1].toordinal()))
        ax1.set_xlabel(x_axis_label)
        # ax2.set_ylabel('Power [p.u.]') if in_per_unit else ax2.set_ylabel('Power [W]')
        # ax2.set_ylim((0.0, 1.0)) if in_per_unit else ax2.set_ylim((0.0, 30.0))
        h1, l1 = ax1.get_legend_handles_labels()
        # h2, l2 = ax2.get_legend_handles_labels()
        lax.legend((h1), (l1), borderaxespad=0, prop={'size': legend_font_size})
        lax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'{der_name}_{control}.png'))
        plt.show()
        plt.close()

    for state in bg.states:
        # Create plot.
        fig, (ax1, lax) = plt.subplots(ncols=2, figsize=figsize, gridspec_kw={"width_ratios": [100, 1]})
        ax1.set_title(f'DER {der_name}')
        ax1.plot(
            results['state_vector'][state],
            label='Optimal',
            drawstyle='steps-post',
            color=colors[0],
            linewidth=linewidth)
        ax1.grid(show_grid)
        ax1.set_xlabel(x_axis_label)
        ax1.set_ylabel(f'{state}')

        # ax2 = plt.twinx(ax1)
        ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(x_label_date_format))
        ax1.set_xlim((bg.timesteps[0].toordinal(), bg.timesteps[-1].toordinal()))
        ax1.set_xlabel(x_axis_label)
        # ax2.set_ylabel('Power [p.u.]') if in_per_unit else ax2.set_ylabel('Power [W]')
        # ax2.set_ylim((0.0, 1.0)) if in_per_unit else ax2.set_ylim((0.0, 30.0))
        h1, l1 = ax1.get_legend_handles_labels()
        # h2, l2 = ax2.get_legend_handles_labels()
        lax.legend((h1), (l1), borderaxespad=0, prop={'size': legend_font_size})
        lax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'{der_name}_{state}.png'))
        plt.show()
        plt.close()


    if price_timeseries is not None:

        profit_vector = pd.DataFrame(0.0, index=bg.timesteps,
                                     columns=pd.Index(['profit_value']))

        for timestep in bg.timesteps:
            profit_vector.at[timestep, 'profit_value'] = (
                    price_timeseries.at[timestep, 'price_value']
                    # revenue from selling power
                    * sum(
                        results['output_vector'].at[timestep, output]
                        for output in bg.outputs if output is 'active_power'
                        ) * bg.timestep_interval.seconds / 3600
                        -
                        # marginal costs of power generation
                        bg.marginal_cost
                        * sum(
                            results['output_vector'].at[timestep, output]
                            for output in bg.outputs if 'active_power' in output and 'CHP' in output
                    ) * bg.timestep_interval.seconds / 3600
            )

        # Create plot.
        fig, (ax1, lax) = plt.subplots(ncols=2, figsize=figsize, gridspec_kw={"width_ratios": [100, 1]})
        ax1.set_title(f'DER {der_name} - Profit and Energy Price per time interval')
        ax1.plot(
            profit_vector['profit_value'] * 1000,
            label='Profit at time step',
            drawstyle='steps-post',
            color=colors[0],
            linewidth=linewidth)
        ax1.grid(show_grid)
        ax1.set_ylabel('Profit (EUR)')
        ax2 = plt.twinx(ax1)
        ax2.plot(
            price_timeseries['price_value'] * 1000,
            label='Market Price (EUR/kWh)',
            drawstyle='steps-post',
            color=colors[1],
            linewidth=linewidth)
        ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(x_label_date_format))
        ax1.set_xlim((bg.timesteps[0].toordinal(), bg.timesteps[-1].toordinal()))
        ax1.set_xlabel(x_axis_label)
        ax2.set_ylabel(f'{price_timeseries["price_type"][0]} (EUR/kWh)')
        #ax2.set_ylim((0.0, 1.0)) if in_per_unit else ax2.set_ylim((0.0, 30.0)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        lax.legend((*h1, *h2), (*l1, *l2), borderaxespad=0, prop={'size': legend_font_size})
        lax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'{der_name}_Profit-and-Energy-Price.png'))
        plt.show()
        plt.close()

        print("Total profit in euros:", sum(profit_vector['profit_value']))

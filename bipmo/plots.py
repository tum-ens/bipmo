"""Plots module for the bipmo flexible biogas plant model"""

import matplotlib.pyplot as plt
import pandas as pd

import fledge.data_interface
import fledge.der_models


# Plot results.
def generate_biogas_plant_plots(
        results: fledge.data_interface.ResultsDict,
        flexible_biogas_plant_model: fledge.der_models.FlexibleBiogasPlantModel,
        price_timeseries: pd.DataFrame = None
):

    for output in flexible_biogas_plant_model.outputs:
        plt.plot(flexible_biogas_plant_model.output_maximum_timeseries[output], label="Maximum", drawstyle='steps-post')
        plt.plot(flexible_biogas_plant_model.output_minimum_timeseries[output], label="Minimum", drawstyle='steps-post')
        plt.plot(results['output_vector'][output], label="Optimal", drawstyle='steps-post')
        plt.legend()
        plt.xlabel('Timesteps (day and hour)')
        plt.title(f"Output: {output}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

    for control in flexible_biogas_plant_model.controls:
        plt.plot(results['control_vector'][control], label="Optimal", drawstyle='steps-post', color='#D55E00')
        plt.legend()
        plt.xlabel('Timesteps (day and hour)')
        plt.title(f"Control: {control}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

    for state in flexible_biogas_plant_model.states:
        plt.plot(results['state_vector'][state], label="Optimal", drawstyle='steps-post', color='#D55E00')
        plt.legend()
        plt.xlabel('Timesteps (day and hour)')
        plt.title(f"State: {state}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

    if price_timeseries is not None:

        profit_vector = pd.DataFrame(0.0, index=flexible_biogas_plant_model.timesteps,
                                     columns=pd.Index(['profit_value']))

        for timestep in flexible_biogas_plant_model.timesteps:
            profit_vector.at[timestep, 'profit_value'] = (
                    price_timeseries.at[timestep, 'price_value']
                    # revenue from selling power
                    * sum(
                        results['output_vector'].at[timestep, output]
                        for output in flexible_biogas_plant_model.outputs if output is 'active_power'
                        ) * flexible_biogas_plant_model.timestep_interval.seconds / 3600
                    -
                    # marginal costs of power generation
                    flexible_biogas_plant_model.marginal_cost
                    * sum(
                        results['output_vector'].at[timestep, output]
                        for output in flexible_biogas_plant_model.outputs if 'active_power' in output and 'CHP' in output
                    ) * flexible_biogas_plant_model.timestep_interval.seconds / 3600
            )

        plt.plot(profit_vector['profit_value'], label="Optimal", drawstyle='steps-post', color='#D55E00')
        plt.legend()
        plt.title('Profit per time interval (euros)')
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.xlabel('Timesteps (day and hour)')
        plt.ylabel('Profit (euros)')
        plt.show()
        plt.close()

        plt.plot(price_timeseries['price_value'], drawstyle='steps-post')
        plt.title(f"Price in EUR per Wh: {price_timeseries['price_type'][0]}")
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
                 'rotation', 45)
        plt.show()
        plt.close()

        print("Total profit in euros:", sum(profit_vector['profit_value']))

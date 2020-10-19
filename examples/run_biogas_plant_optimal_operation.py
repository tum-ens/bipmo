"""
Example script for setting up and solving a flexible biogas plant optimal operation problem.
Biogas plant not connected to the grid
"""

import pyomo.environ as pyo
import pandas as pd
from datetime import datetime
import os

import bipmo.biogas_plant_models
import bipmo.plots

# Settings.
scenario_name = 'biogas_plant_1'
solver_name = 'gurobi'
plots = True  # If True, script may produce plots.
run_milp = True  # if True, script will formulate a MILP and then use the results as integers in a second run

# obtain base_path
base_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))

# Obtain results path and create directory
results_path = os.path.join(base_path, 'results', scenario_name + '_' + datetime.now().strftime("%d-%m-%Y_%H-%M"))
os.mkdir(results_path)

# Obtain scenario data
plant_scenarios = pd.read_csv(
        os.path.join(base_path, 'data/biogas_plant_scenario.csv')
    )
plant_scenarios = plant_scenarios[
    plant_scenarios['scenario_name'] == scenario_name]
plant_scenarios.index = pd.Index([scenario_name])

# Obtain price timeseries.
price_type = 'EPEX SPOT Power DE Day Ahead'

price_timeseries = pd.read_csv(
    os.path.join(base_path, 'data/price_timeseries.csv')
)
price_timeseries = price_timeseries[
    price_timeseries['price_type'] == price_type]

price_timeseries['time'] = pd.to_datetime(price_timeseries['time'])

time_start = plant_scenarios['time_start'][0]
time_end = plant_scenarios['time_end'][0]
price_timeseries = price_timeseries[
    (price_timeseries['time'] >= time_start) &
    (price_timeseries['time'] <= time_end)]
# rename the columns
price_timeseries.columns = ['price_type', 'timestep', 'price_value']
# make timestep column the index
price_timeseries = price_timeseries.set_index('timestep')

# Instantiate an empty chp_schedule
chp_schedule: pd.DataFrame

for i in range(2):
    if run_milp:
        if i == 0:
            is_milp = True
        else:
            is_milp = False
    else:
        is_milp = False
        i = 3  # will stop from iterating again

    # Get the biogas plant model and set the switches flag accordingly
    flexible_biogas_plant_model = bipmo.biogas_plant_models.FlexibleBiogasPlantModel(scenario_name)

    if (not is_milp) and run_milp:
        # set the chp_schedule resulting from the milp optimization
        flexible_biogas_plant_model.chp_schedule = chp_schedule

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    flexible_biogas_plant_model.define_optimization_variables(optimization_problem)
    flexible_biogas_plant_model.define_optimization_constraints(optimization_problem)

    if is_milp:
        # define binary variables for MILP solution
        optimization_problem.binary_variables = pyo.Var(flexible_biogas_plant_model.timesteps,
                                                        [flexible_biogas_plant_model.der_name],
                                                        flexible_biogas_plant_model.switches,
                                                        domain=pyo.Binary)

        for timestep in flexible_biogas_plant_model.timesteps:
            for output in flexible_biogas_plant_model.outputs:
                if 'active_power_Wel' in output:
                    for chp in flexible_biogas_plant_model.CHP_list:
                        if chp in output and any(flexible_biogas_plant_model.switches.str.contains(chp)):
                            optimization_problem.der_model_constraints.add(
                                optimization_problem.output_vector[timestep, flexible_biogas_plant_model.der_name, output]
                                >=
                                flexible_biogas_plant_model.output_minimum_timeseries.at[timestep, output]
                                * optimization_problem.binary_variables[timestep, flexible_biogas_plant_model.der_name, chp + '_switch']
                            )
                            optimization_problem.der_model_constraints.add(
                                optimization_problem.output_vector[timestep, flexible_biogas_plant_model.der_name, output]
                                <=
                                flexible_biogas_plant_model.output_maximum_timeseries.at[timestep, output]
                                * optimization_problem.binary_variables[timestep, flexible_biogas_plant_model.der_name, chp + '_switch']
                            )
    else:  # define the constraints without the binary variables
        for timestep in flexible_biogas_plant_model.timesteps:
            for output in flexible_biogas_plant_model.outputs:
                if flexible_biogas_plant_model.chp_schedule is not None and 'active_power_Wel' in output:
                    for chp in flexible_biogas_plant_model.CHP_list:
                        if chp in output and any(flexible_biogas_plant_model.switches.str.contains(chp)):
                            optimization_problem.der_model_constraints.add(
                                optimization_problem.output_vector[timestep, flexible_biogas_plant_model.der_name, output]
                                >=
                                flexible_biogas_plant_model.output_minimum_timeseries.at[timestep, output]
                                * flexible_biogas_plant_model.chp_schedule.loc[timestep, chp+'_switch']
                            )
                            optimization_problem.der_model_constraints.add(
                                optimization_problem.output_vector[timestep, flexible_biogas_plant_model.der_name, output]
                                <=
                                flexible_biogas_plant_model.output_maximum_timeseries.at[timestep, output]
                                * flexible_biogas_plant_model.chp_schedule.loc[timestep, chp+'_switch']
                            )

    # Define the optimization objective with the price timeseries
    flexible_biogas_plant_model.define_optimization_objective(optimization_problem, price_timeseries)

    # Solve optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=True)
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")

    if is_milp:
        # get the MILP solution for the biogas plant schedule
        binaries = optimization_problem.binary_variables
        timesteps = flexible_biogas_plant_model.timesteps
        chp_schedule = flexible_biogas_plant_model.chp_schedule
        for timestep in timesteps:
            for chp in flexible_biogas_plant_model.CHP_list:
                chp_schedule.loc[timestep, chp+'_switch'] = \
                    binaries[timestep, flexible_biogas_plant_model.der_name, chp+'_switch'].value


results = flexible_biogas_plant_model.get_optimization_results(optimization_problem)

print(results)

if plots:
    bipmo.plots.generate_biogas_plant_plots(results, flexible_biogas_plant_model, results_path, price_timeseries)


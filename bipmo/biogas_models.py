"""Based on fledge/fledge/der_models.py"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import datetime as dt

import bipmo.bipmo.biogas_plant_model
import fledge.config
import fledge.data_interface
import fledge.electric_grid_models
import fledge.thermal_grid_models
import fledge.utils

logger = fledge.config.get_logger(__name__)


class DERModel(object):
    """DER model object."""

    der_name: str
    timesteps: pd.Index
    active_power_nominal_timeseries: pd.Series
    reactive_power_nominal_timeseries: pd.Series
    thermal_power_nominal_timeseries: pd.Series


class FlexibleBiogasModel(DERModel):
    """Flexible Biogas plant model.
    This is the equivalent to FlexibleDERModel in der_models in FLEDGE
    TODO: this should be removed and moved into der_models
    """

    states: pd.Index
    controls: pd.Index
    outputs: pd.Index
    state_vector_initial: pd.Series
    state_matrix: pd.DataFrame
    control_matrix: pd.DataFrame
    disturbance_matrix: pd.DataFrame
    state_output_matrix: pd.DataFrame
    control_output_matrix: pd.DataFrame
    output_maximum_timeseries: pd.DataFrame
    output_minimum_timeseries: pd.DataFrame

    def define_optimization_variables(
            self,
            optimization_problem: pyo.ConcreteModel,
    ):

        # Define variables.
        optimization_problem.state_vector = pyo.Var(self.timesteps, [self.der_name], self.states)
        optimization_problem.control_vector = pyo.Var(self.timesteps, [self.der_name], self.controls)
        optimization_problem.output_vector = pyo.Var(self.timesteps, [self.der_name], self.outputs)

    def define_optimization_constraints(
        self,
        optimization_problem: pyo.ConcreteModel,
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault = None,
        power_flow_solution: fledge.electric_grid_models.PowerFlowSolution = None,
        thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel = None,
        thermal_power_flow_solution: fledge.thermal_grid_models.ThermalPowerFlowSolution = None
    ):

        # Define constraints.
        if optimization_problem.find_component('der_model_constraints') is None:
            optimization_problem.der_model_constraints = pyo.ConstraintList()

        # Initial state.
        for state in self.states:
            optimization_problem.der_model_constraints.add(
                optimization_problem.state_vector[self.timesteps[0], self.der_name, state]
                ==
                self.state_vector_initial.at[state]
            )

        for timestep in self.timesteps[:-1]:

            # State equation.
            for state in self.states:
                optimization_problem.der_model_constraints.add(
                    optimization_problem.state_vector[timestep + self.timestep_interval, self.der_name, state]
                    ==
                    sum(
                        self.state_matrix.at[state, state_other]
                        * optimization_problem.state_vector[timestep, self.der_name, state_other]
                        for state_other in self.states
                    )
                    + sum(
                        self.control_matrix.at[state, control]
                        * optimization_problem.control_vector[timestep, self.der_name, control]
                        for control in self.controls
                    )
                    + sum(
                        self.disturbance_matrix.at[state, disturbance]
                        * self.disturbance_timeseries.at[timestep, disturbance]
                        for disturbance in self.disturbances
                    )
                )

        for timestep in self.timesteps:

            # Output equation.
            for output in self.outputs:
                optimization_problem.der_model_constraints.add(
                    optimization_problem.output_vector[timestep, self.der_name, output]
                    ==
                    sum(
                        self.state_output_matrix.at[output, state]
                        * optimization_problem.state_vector[timestep, self.der_name, state]
                        for state in self.states
                    )
                    + sum(
                        self.control_output_matrix.at[output, control]
                        * optimization_problem.control_vector[timestep, self.der_name, control]
                        for control in self.controls
                    )
                    + sum(
                        self.disturbance_output_matrix.at[output, disturbance]
                        * self.disturbance_timeseries.at[timestep, disturbance]
                        for disturbance in self.disturbances
                    )
                )

        # Output limits.
        for timestep in self.timesteps:
            for output in self.outputs:
                optimization_problem.der_model_constraints.add(
                    optimization_problem.output_vector[timestep, self.der_name, output]
                    >=
                    self.output_minimum_timeseries.at[timestep, output]
                )
                optimization_problem.der_model_constraints.add(
                    optimization_problem.output_vector[timestep, self.der_name, output]
                    <=
                    self.output_maximum_timeseries.at[timestep, output]
                )

            # Feedstock input limits (maximum daily or hourly feed-in depending on available feedstock).
            for control in self.controls:
                if self.feedstock_limit_type == 'daily':
                    if ('mass_flow' in control) and (timestep + dt.timedelta(days=1) - self.timestep_interval <= self.timestep_end):
                        optimization_problem.der_model_constraints.add(
                            sum(
                                self.timestep_interval.seconds *
                                optimization_problem.control_vector[timestep + i * self.timestep_interval, self.der_name, control]
                                for i in range(int(dt.timedelta(days=1)/self.timestep_interval))
                            )
                            <= self.available_feedstock*1000/365
                        )
                elif self.feedstock_limit_type == 'hourly':
                    if ('mass_flow' in control) and (timestep + dt.timedelta(hours=1) - self.timestep_interval <= self.timestep_end):
                        optimization_problem.der_model_constraints.add(
                            sum(
                                self.timestep_interval.seconds *
                                optimization_problem.control_vector[
                                    timestep + i * self.timestep_interval, self.der_name, control]
                                for i in range(int(dt.timedelta(hours=1) / self.timestep_interval))
                            )
                            <= self.available_feedstock * 1000 / (365*24)
                        )

        # CHP Ramp rate constraints.
        for timestep in self.timesteps[:-1]:
            for output in self.outputs:
                for i in self.CHP_list:
                    if ('active_power' in output) and (i in output):
                        optimization_problem.der_model_constraints.add(
                            optimization_problem.output_vector[timestep + self.timestep_interval, self.der_name, output]
                            - optimization_problem.output_vector[timestep, self.der_name, output]
                            <=
                            self.ramp_rate_list.loc[i, 'ramp_rate_W_min'] * self.timestep_interval.seconds/60
                        )
                        optimization_problem.der_model_constraints.add(
                            optimization_problem.output_vector[timestep + self.timestep_interval, self.der_name, output]
                            - optimization_problem.output_vector[timestep, self.der_name, output]
                            >=
                            - self.ramp_rate_list.loc[i, 'ramp_rate_W_min'] * self.timestep_interval.seconds/60
                        )

        # Final SOC storage
        if self.SOC_end == 'init':
            # Final SOC greater or equal to initial SOC
            optimization_problem.der_model_constraints.add(
                optimization_problem.output_vector[self.timesteps[-1], self.der_name, self.scenario_name
                                                   + '_storage_content_m3']
                >= self.state_vector_initial[self.scenario_name + '_storage_content_m3']
            )
        elif self.SOC_end == 'min':
            # Minimal SOC reached at the last step
             optimization_problem.der_model_constraints.add(
                 optimization_problem.output_vector[self.timesteps[-1], self.der_name, self.scenario_name + '_storage_content_m3']
                 >= self.SOC_min
             )

        # Define connection constraints.
        if electric_grid_model is not None:
            der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=self.der_name))
            der = electric_grid_model.ders[der_index]

            for timestep in self.timesteps:
                optimization_problem.der_model_constraints.add(
                    optimization_problem.der_active_power_vector_change[timestep, der]
                    ==
                    sum(
                        optimization_problem.output_vector[timestep, self.der_name, output]
                        for output in self.outputs if 'active' in output
                    )
                    - np.real(
                        power_flow_solution.der_power_vector[der_index]
                              )
                )
                optimization_problem.der_model_constraints.add(
                    optimization_problem.der_reactive_power_vector_change[timestep, der]
                    ==
                    sum(
                        optimization_problem.output_vector[timestep, self.der_name, output]
                        for output in self.outputs if 'react' in output
                    )
                    - np.imag(
                        power_flow_solution.der_power_vector[der_index]
                    )
                )

    def define_optimization_objective(
            self,
            optimization_problem: pyo.ConcreteModel,
            price_timeseries: pd.DataFrame,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault = None,
            thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel = None,
    ):

        # Define objective.
        if optimization_problem.find_component('objective') is None:
            optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.maximize)

        optimization_problem.objective.expr += (
            sum(
                price_timeseries.at[timestep, 'price_value']/1000000
                * (
                    # Income from selling power.
                    sum(
                        optimization_problem.output_vector[timestep, self.der_name, output]
                        for output in self.outputs if 'active_power' in output
                    ) * self.timestep_interval.seconds / 3600
                    -
                    # Power requirements.
                    sum(
                        optimization_problem.output_vector[timestep, self.der_name, output]
                        for output in self.outputs if 'act_power_own_consumption' in output
                    ) * self.timestep_interval.seconds / 3600
                )
                -
                # Substrate costs.
                self.feedstock_cost
                * sum(
                    optimization_problem.output_vector[timestep, self.der_name, output]
                    for output in self.outputs if 'active_power' in output
                    ) * self.timestep_interval.seconds / 3600
                for timestep in self.timesteps
            )
        )

    def get_optimization_results(
            self,
            optimization_problem: pyo.ConcreteModel,
            price_timeseries: pd.DataFrame
    ) -> fledge.data_interface.ResultsDict:

        # Instantiate results variables.
        state_vector = pd.DataFrame(0.0, index=self.timesteps, columns=self.states)
        control_vector = pd.DataFrame(0.0, index=self.timesteps, columns=self.controls)
        output_vector = pd.DataFrame(0.0, index=self.timesteps, columns=self.outputs)
        profit_vector = pd.DataFrame(0.0, index=self.timesteps, columns=pd.Index(['profit_value']))

        # Obtain results.
        for timestep in self.timesteps:
            for state in self.states:
                state_vector.at[timestep, state] = (
                    optimization_problem.state_vector[timestep, self.der_name, state].value
                )
            for control in self.controls:
                control_vector.at[timestep, control] = (
                    optimization_problem.control_vector[timestep, self.der_name, control].value
                )
            for output in self.outputs:
                output_vector.at[timestep, output] = (
                    optimization_problem.output_vector[timestep, self.der_name, output].value
                )

            profit_vector.at[timestep, 'profit_value'] = (
                price_timeseries.at[timestep, 'price_value']/1000000
                * (
                    # Income from selling power
                    sum(
                        output_vector.at[timestep, output]
                        for output in self.outputs if 'active_power' in output
                        ) * self.timestep_interval.seconds / 3600
                    -
                    # Power requirements
                    sum(
                        output_vector.at[timestep, output]
                        for output in self.outputs if 'act_power_own_consumption' in output
                        ) * self.timestep_interval.seconds / 3600
                )
                -
                # Substrate costs
                self.feedstock_cost
                * sum(
                    output_vector.at[timestep, output]
                    for output in self.outputs if 'active_power' in output
                    ) * self.timestep_interval.seconds / 3600
            )

        return fledge.data_interface.ResultsDict(
            state_vector=state_vector,
            control_vector=control_vector,
            output_vector=output_vector,
            profit_vector=profit_vector
        )


class FlexibleBiogasPlantModel(FlexibleBiogasModel):
    """Flexible Biogas plant model object."""

    power_factor_nominal: np.float
    is_electric_grid_connected: np.bool

    def __init__(
            self,
            der_data: fledge.data_interface.DERData,
            der_name: str
    ):

        # Store DER name.
        self.der_name = der_name

        # Obtain biogas data by `der_name`.
        biogas_plant = der_data.biogas_plants.loc[der_name, :]

        # Store biogas scenario name
        self.scenario_name = biogas_plant['model_name']

        # Obtain grid connection flags.
        self.is_electric_grid_connected = pd.notnull(biogas_plant.at['electric_grid_name'])

        # Obtain bipmo biogas plant model.
        flexible_biogas_plant_model = (
            bipmo.bipmo.biogas_plant_model.BiogasModel(
                biogas_plant.at['model_name'],
                timestep_start=der_data.scenario_data.scenario.at['timestep_start'],
                timestep_end=der_data.scenario_data.scenario.at['timestep_end'],
                timestep_interval=der_data.scenario_data.scenario.at['timestep_interval'],
            )
        )

        # Store timesteps.
        self.timesteps = flexible_biogas_plant_model.timesteps
        self.timestep_interval = flexible_biogas_plant_model.timestep_interval
        self.timestep_end = flexible_biogas_plant_model.timestep_end

        # Construct nominal active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
                pd.Series(1.0, index=self.timesteps, name='active_power')
                * (
                    biogas_plant.at['active_power_nominal']
                    if pd.notnull(biogas_plant.at['active_power_nominal'])
                    else 0.0
                )
        )
        self.reactive_power_nominal_timeseries = (
                pd.Series(1.0, index=self.timesteps, name='reactive_power')
                * (
                    biogas_plant.at['reactive_power_nominal']
                    if pd.notnull(biogas_plant.at['reactive_power_nominal'])
                    else 0.0
                )
        )

        # Obtain indexes.
        self.states = flexible_biogas_plant_model.states
        self.controls = flexible_biogas_plant_model.controls
        self.outputs = flexible_biogas_plant_model.outputs
        self.disturbances = flexible_biogas_plant_model.disturbances

        # Obtain ramp information
        self.CHP_list = flexible_biogas_plant_model.CHP_list
        self.elec_cap_list = flexible_biogas_plant_model.elec_cap_list
        self.ramp_rate_list = flexible_biogas_plant_model.ramp_rate_list

        # Obtain digester information
        self.time_constant = flexible_biogas_plant_model.a1
        self.feedstock_cost = flexible_biogas_plant_model.plant_feedstock.loc[self.scenario_name, 'cost_feedstock_euro_Wh']
        self.feedstock_limit_type = flexible_biogas_plant_model.plant_scenarios.loc[
            self.scenario_name, 'availability_limit_type']
        self.available_feedstock = flexible_biogas_plant_model.plant_scenarios.loc[self.scenario_name, 'availability_substrate_ton_per_year']

        # Obtain storage information
        self.SOC_end = flexible_biogas_plant_model.plant_storage.loc[self.scenario_name, 'SOC_end']
        self.SOC_min = flexible_biogas_plant_model.plant_storage.loc[self.scenario_name, 'SOC_min_m3']

        # Obtain initial state.
        self.state_vector_initial = flexible_biogas_plant_model.state_vector_initial

        # Obtain state space matrices.
        self.state_matrix = flexible_biogas_plant_model.state_matrix
        self.control_matrix = flexible_biogas_plant_model.control_matrix
        self.disturbance_matrix = flexible_biogas_plant_model.disturbance_matrix
        self.state_output_matrix = flexible_biogas_plant_model.state_output_matrix
        self.control_output_matrix = flexible_biogas_plant_model.control_output_matrix
        self.disturbance_output_matrix = flexible_biogas_plant_model.disturbance_output_matrix

        # Obtain disturbance timeseries
        self.disturbance_timeseries = flexible_biogas_plant_model.disturbance_timeseries

        # Obtain output constraint timeseries.
        self.output_maximum_timeseries = flexible_biogas_plant_model.output_constraint_timeseries_maximum
        self.output_minimum_timeseries = flexible_biogas_plant_model.output_constraint_timeseries_minimum

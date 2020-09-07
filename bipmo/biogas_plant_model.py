import numpy as np
import pandas as pd
import scipy.linalg
import os


class BiogasModel(object):
    """ Create a linear state-space model for a biogas power plant, given data about the plant.
    The class defines and creates, among other attributes, the discretized state, control, state_output and
    control_output matrices, necessary to the numerical resolution of the model.
    Based on cobmo/building_model.py"""
    scenario_name: str
    states: pd.Index
    controls: pd.Index
    outputs: pd.Index
    disturbances: pd.Index
    state_vector_initial: pd.Series
    state_matrix: pd.DataFrame
    control_matrix: pd.DataFrame
    disturbance_matrix: pd.DataFrame
    state_output_matrix: pd.DataFrame
    control_output_matrix: pd.DataFrame
    disturbance_output_matrix: pd.DataFrame
    timestep_start: pd.Timestamp
    timestep_end: pd.Timestamp
    timestep_interval: pd.Timedelta
    timesteps: pd.Index
    disturbance_timeseries: pd.DataFrame
    output_constraint_timeseries_maximum: pd.DataFrame
    output_constraint_timeseries_minimum: pd.DataFrame

    def __init__(
            self,
            scenario_name: str,
            timestep_start=None,
            timestep_end=None,
            timestep_interval=None,
            connect_electric_grid=True,
    ):

        # Scenario name.
        self.scenario_name = scenario_name

        # Define the biogas plant model (change paths accordingly).
        base_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))

        # Load the scenario.
        self.plant_scenarios = pd.read_csv(
            os.path.join(base_path, 'data/biogas_plant_scenario.csv')
                                           )
        self.plant_scenarios = self.plant_scenarios[
            self.plant_scenarios['scenario_name'] == self.scenario_name]
        self.plant_scenarios.index = pd.Index([self.scenario_name])

        # Load feedstock data used in the scenario.
        self.plant_feedstock = pd.read_csv(
            os.path.join(base_path, 'data/biogas_plant_feedstock.csv')
                                           )
        self.plant_feedstock = self.plant_feedstock[
            self.plant_feedstock['feedstock_type']
            == self.plant_scenarios.loc[self.scenario_name, 'feedstock_type']
            ]
        self.plant_feedstock.index = pd.Index([self.scenario_name])

        # Load CHP data used in the scenario.
        self.CHP_list = self.plant_scenarios.CHP_name[self.scenario_name].split()
        self.number_CHP = len(self.CHP_list)
        self.plant_CHP_source = pd.read_csv(
            os.path.join(base_path, 'data/biogas_plant_CHP.csv')
                                            )
        self.plant_CHP = pd.DataFrame(columns=self.plant_CHP_source.columns)
        for i in self.CHP_list:
            self.plant_CHP = pd.concat([
                self.plant_CHP,
                self.plant_CHP_source[self.plant_CHP_source['CHP_name'] == i]
            ])
        self.plant_CHP.index = self.plant_CHP['CHP_name']
        self.elec_cap_list = pd.DataFrame([cap for cap in self.plant_CHP.elec_cap_Wel],
                                          index=self.CHP_list,
                                          columns=['elec_cap_Wel'])
        self.ramp_rate_list = pd.DataFrame([rate for rate in self.plant_CHP.ramp_capacity_W_min],
                                           index=self.CHP_list,
                                           columns=['ramp_rate_W_min'])

        # Load storage data used in the scenario.
        self.plant_storage = pd.read_csv(
            os.path.join(base_path, 'data/biogas_plant_storage.csv')
                                         )
        self.plant_storage = self.plant_storage[
            self.plant_storage['storage_name']
            == self.plant_scenarios.loc[self.scenario_name, 'storage_name']
            ]
        self.plant_storage.index = pd.Index([self.scenario_name])

        # Define sets.

        # State variables.
        self.states = pd.Index(
            pd.concat([
                # Biogas production rate from the digester.
                self.plant_scenarios['scenario_name']
                + '_prod_biogas_m3_s-1',
                # Storage biogas content.
                self.plant_scenarios['scenario_name']
                + '_storage_content_m3'
            ]),
            name='state_name'
        )

        # Control variables.
        self.controls = pd.Index(
            # Feedstock mass flow.
            'mass_flow_kg_s-1_'
            + self.plant_scenarios['feedstock_type'],
            name='control_name'
        )
        for i in range(len(self.CHP_list)):
            self.controls = pd.Index([
                # CHPs Biogas inflows
                self.plant_CHP['CHP_name'][i] + '_biogas_volume_inflow_m3_s-1'
                ]).union(self.controls)

            # Output variables.
        self.outputs = pd.Index(
            pd.concat([
                # Biogas production rate from the digester / Biogas inflow to the storage.
                self.plant_scenarios['scenario_name']
                + '_prod_biogas_m3_s-1',
                # Active power requirement of the digester.
                self.plant_scenarios['scenario_name']
                + '_act_power_own_consumption_Wel',
                # Heat requirement of the digester.
                self.plant_scenarios['scenario_name']
                + '_heat_own_consumption_Wth',
                # Storage biogas content.
                self.plant_scenarios['scenario_name']
                + '_storage_content_m3'
            ]),
            name='output_name'
        )
        for i in range(len(self.CHP_list)):
            self.outputs = pd.Index([
                # CHPs active power production.
                self.plant_CHP['CHP_name'][i] + '_active_power_Wel',
                # CHPs reactive power production.
                self.plant_CHP['CHP_name'][i] + '_react_power_Var',
                # CHPs heat power production.
                self.plant_CHP['CHP_name'][i] + '_heat_Wth'
                ]).union(self.outputs)

            # Disturbance variables (empty).
        self.disturbances = pd.Index([None],
                                     name='disturbance_name')

        # Define timesteps.
        if timestep_start is not None:
            self.timestep_start = pd.Timestamp(timestep_start)
        else:
            self.timestep_start = pd.Timestamp(self.plant_scenarios.loc[self.scenario_name, 'time_start'])
        if timestep_end is not None:
            self.timestep_end = pd.Timestamp(timestep_end)
        else:
            self.timestep_end = pd.Timestamp(self.plant_scenarios.loc[self.scenario_name, 'time_end'])
        if timestep_interval is not None:
            self.timestep_interval = pd.Timedelta(timestep_interval)
        else:
            self.timestep_interval = pd.Timedelta(self.plant_scenarios.loc[self.scenario_name, 'time_step'])
        self.timesteps = pd.Index(
            pd.date_range(
                start=self.timestep_start,
                end=self.timestep_end,
                freq=self.timestep_interval
            ),
            name='time'
        )

        # Instantiate empty state-space model matrices.
        self.state_matrix = pd.DataFrame(
            0.0,
            self.states,
            self.states
        )
        self.control_matrix = pd.DataFrame(
            0.0,
            self.states,
            self.controls
        )
        self.disturbance_matrix = pd.DataFrame(
            0.0,
            self.states,
            self.disturbances
        )
        self.state_output_matrix = pd.DataFrame(
            0.0,
            self.outputs,
            self.states
        )
        self.control_output_matrix = pd.DataFrame(
            0.0,
            self.outputs,
            self.controls
        )
        self.disturbance_output_matrix = pd.DataFrame(
            0.0,
            self.outputs,
            self.disturbances
        )

        # Define useful values.
        lhv_table = pd.DataFrame(
            # Lower heating value of methane in J/m3.
            [35.8e6],
            pd.Index(['LHV_methane']),
            pd.Index(['LHV value (in J/m^3)'])
        )
        temp_in = self.plant_scenarios.loc[
            # Temperature of the digestion process in °C.
            self.scenario_name, 'digester_temp']

        cp_water = 4182  # Specific heat of water in J/(K*kg) at 20°C.

        # Define coefficients

            # Define the heat and power requirements coefficients.

        # Heat requirement to increase the introduced feedstock's temperature
        # to the operating temperature in J/kg.
        self.gain_parasitic_heat = cp_water * (
                temp_in - self.plant_scenarios.loc[self.scenario_name, 'temperature_outside']
        )
        # Power requirement for the stirring and pumping of the feedstock in Je/kg
        # (value for cattle slurry, kept for other feedstock types).
        self.gain_parasitic_power = 25920

            # Define the heat and power CHP coefficients.
        self.set_gains = pd.Index([])
        for i in range(len(self.CHP_list)):
            self.set_gains = pd.Index([
                self.plant_CHP['CHP_name'][i] + '_biogas_volume_inflow_m3_s-1'
                ]).union(self.set_gains)

        self.gain_heat = pd.DataFrame(
            0.0,
            pd.Index([0]),
            pd.Index(range(0, self.set_gains.size))
        )
        self.gain_power = pd.DataFrame(
            0.0,
            pd.Index([0]),
            pd.Index(range(0, self.set_gains.size))
        )

        for i in range(0, self.number_CHP):
            for j in range(0, lhv_table.size):
                self.gain_heat[lhv_table.size * i + j] = self.plant_CHP['therm_eff'][i] * \
                                            lhv_table['LHV value (in J/m^3)'][j] * \
                                            self.plant_feedstock['methane_content'][self.scenario_name]
                self.gain_power[lhv_table.size * i + j] = self.plant_CHP['elec_eff'][i] * \
                                            lhv_table['LHV value (in J/m^3)'][j] * \
                                            self.plant_feedstock['methane_content'][self.scenario_name]

        self.gain_heat.columns = self.set_gains
        self.gain_power.columns = self.set_gains

        # Define the matrices of the system.

            # Define the state matrix.
        self.a1 = (float(self.plant_feedstock.loc[self.scenario_name, 'time_constant_h']) * 3600) ** (-1)
        self.state_matrix.loc[self.scenario_name + '_prod_biogas_m3_s-1', self.scenario_name + '_prod_biogas_m3_s-1'] \
            = -self.a1
        self.state_matrix.loc[self.scenario_name + '_storage_content_m3', self.scenario_name + '_prod_biogas_m3_s-1'] \
            = 1.0

            # Define the control matrix.
        self.b1 = (self.plant_feedstock.loc[self.scenario_name, 'biogas_yield_m3_kgVS-1'])\
             * self.a1 \
             * self.plant_feedstock.loc[self.scenario_name, 'VS']\
             * self.plant_feedstock.loc[self.scenario_name, 'DM']
        for state in self.states:
            for control in self.controls:
                if ('prod' in state) and ('mass_flow' in control):
                    self.control_matrix.loc[state, control] = self.b1
                if ('storage' in state) and ('biogas' in control):
                    self.control_matrix.loc[state, control] = -1.0

            # Define the state output matrix.
        for state in self.states:
            for output in self.outputs:
                if state == output:
                    self.state_output_matrix.loc[state, output] = 1

            # Define the control output matrix.
        for i in self.controls:
            for j in self.outputs:
                if ('active_power' in j) and (i[0:5] == j[0:5]):
                    self.control_output_matrix.loc[j, i]\
                        = self.gain_power[i][0] * self.plant_CHP.loc[i[0:5], 'power_factor']
                if ('react_power' in j) and (i[0:5] == j[0:5]):
                    self.control_output_matrix.loc[j, i] \
                        = self.gain_power[i][0] * (1-self.plant_CHP.loc[i[0:5], 'power_factor'])
                if ('heat' in j) and (i[0:5] == j[0:5]):
                    self.control_output_matrix.loc[j, i] \
                        = self.gain_heat[i][0]

        self.control_output_matrix.loc[
            self.scenario_name+'_act_power_own_consumption_Wel',
            'mass_flow_kg_s-1_'+self.plant_scenarios['feedstock_type'][self.scenario_name]]\
            = self.gain_parasitic_power
        self.control_output_matrix.loc[
            self.scenario_name+'_heat_own_consumption_Wth',
            'mass_flow_kg_s-1_'+self.plant_scenarios['feedstock_type'][self.scenario_name]]\
            = self.gain_parasitic_heat

        # Define the initial state (either digester process starting or already active).
        def define_initial_state():
            if self.plant_scenarios.loc[self.scenario_name, 'init_state_digester'] == 'starting':
                self.state_vector_initial = (
                    pd.Series(
                        [0.0, self.plant_storage.loc[self.scenario_name, 'SOC_init_m3']],
                        index=self.states
                    )
                )
            else:
                self.state_vector_initial = (
                    pd.Series(
                        [float(self.plant_scenarios.loc[self.scenario_name, 'init_state_digester']),
                         self.plant_storage.loc[self.scenario_name, 'SOC_init_m3']],
                        index=self.states
                    )
                )

        # Define zero disturbance timeseries.
        def define_disturbance_timeseries():

            # Reindex, interpolate and construct full disturbance timeseries.
            self.disturbance_timeseries = pd.DataFrame(
                0.0,
                self.timesteps,
                self.disturbances
            )

        # Define output and control constraints.

        def define_output_constraint_timeseries():

            # Instantiate constraint timeseries.
            self.output_constraint_timeseries_maximum = pd.DataFrame(
                +1.0 * np.infty,
                self.timesteps,
                self.outputs
            )
            self.output_constraint_timeseries_minimum = pd.DataFrame(
                -1.0 * np.infty,
                self.timesteps,
                self.outputs
            )

            # Minimum constraint for active power outputs.
            for i in self.CHP_list:
                self.output_constraint_timeseries_minimum.loc[
                    :, self.outputs.str.contains(i + '_active_power_Wel')] \
                    = self.plant_CHP.loc[i, 'elec_min_Wel']

            # Maximum constraint for active power outputs.
                self.output_constraint_timeseries_maximum.loc[
                    :, self.outputs.str.contains(i + '_active_power_Wel')] \
                    = self.plant_CHP.loc[i, 'elec_cap_Wel']

            # Minimum constraint for storage content.
            self.output_constraint_timeseries_minimum.loc[
                :, self.outputs.str.contains('_storage')
            ] = self.plant_storage.loc[self.scenario_name, 'SOC_min_m3']

            # Maximum constraint for storage content.
            self.output_constraint_timeseries_maximum.loc[
                :, self.outputs.str.contains('_storage')
            ] = self.plant_storage.loc[self.scenario_name, 'SOC_max_m3']

            # Minimum constraint for own heat and power consumption.
            self.output_constraint_timeseries_minimum.loc[
                :, self.outputs.str.contains('_own_consumption')
            ] = 0.0

        # Discretize the model
        def discretize_model():
            """ In this model, for non-invertibility reason, biogas production rate's and storage volume's equations
            are discretized separately (ie by line)
            Source: https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
            """

            # Discretize biogas production rate' equation.
            state_matrix_part1 = pd.DataFrame(
                -self.a1,
                pd.Index([self.states[0]]),
                pd.Index([self.states[0]])
            )
            control_matrix_part1 = self.control_matrix.loc[self.control_matrix.index == self.scenario_name +'_prod_biogas_m3_s-1']
            state_matrix_discrete_part1 = scipy.linalg.expm(
                state_matrix_part1.values
                * self.timestep_interval.seconds
            )
            control_matrix_discrete_part1 = (
                np.linalg.matrix_power(
                    state_matrix_part1.values,
                    -1
                ).dot(
                    state_matrix_discrete_part1
                    - np.identity(state_matrix_part1.shape[0])
                ).dot(
                    control_matrix_part1.values
                )
            )

            # Discretize storage volume's equation.
            state_matrix_discrete_part2 = [self.timestep_interval.seconds, 1]
            control_matrix_discrete_part2 = []
            for i in range(len(self.CHP_list)):
                control_matrix_discrete_part2.append(-1 * self.timestep_interval.seconds)
            control_matrix_discrete_part2.append(0)

            # Recombine discretized equations in the matrices
            state_matrix_discrete = pd.DataFrame(
                0.0,
                self.states,
                self.states
            )
            control_matrix_discrete = pd.DataFrame(
                0.0,
                self.states,
                self.controls
            )
            state_matrix_discrete.loc[self.scenario_name +'_prod_biogas_m3_s-1', self.scenario_name+'_prod_biogas_m3_s-1'] = state_matrix_discrete_part1.tolist()[0][0]
            state_matrix_discrete.loc[self.scenario_name + '_storage_content_m3', :] = state_matrix_discrete_part2
            control_matrix_discrete.loc[self.scenario_name +'_prod_biogas_m3_s-1', :] = control_matrix_discrete_part1
            control_matrix_discrete.loc[self.scenario_name +'_storage_content_m3', :] = control_matrix_discrete_part2

            self.state_matrix = pd.DataFrame(
                data=state_matrix_discrete,
                index=self.state_matrix.index,
                columns=self.state_matrix.columns
            )
            self.control_matrix = pd.DataFrame(
                data=control_matrix_discrete,
                index=self.control_matrix.index,
                columns=self.control_matrix.columns
            )

        # Apply the functions.
        define_initial_state()
        define_disturbance_timeseries()
        define_output_constraint_timeseries()
        discretize_model()

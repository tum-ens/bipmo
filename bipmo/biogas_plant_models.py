import numpy as np
import pandas as pd
import scipy.linalg
import os
import inspect
import sys


class BiogasPlantModel(object):
    """
    BiogasPlantModel represents all attributes and functions that all biogas plants have in common. It is the basis for
    every model that inherits from it. Caution: It does not work as a standalone model!
    """
    model_type: str = None
    plant_scenarios: pd.DataFrame
    states: pd.Index
    controls: pd.Index
    outputs: pd.Index
    switches: pd.Index
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
    lhv_table: pd.DataFrame
    temp_in: float
    cp_water: float

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

        # Define useful values.
        self.lhv_table = pd.DataFrame(
            # Lower heating value of methane in J/m3.
            [35.8e6],
            pd.Index(['LHV_methane']),
            pd.Index(['LHV value (in J/m^3)'])
        )
        self.temp_in = self.plant_scenarios.loc[
            # Temperature of the digestion process in °C.
            self.scenario_name, 'digester_temp']

        self.cp_water = 4182  # Specific heat of water in J/(K*kg) at 20°C.

        # Define CHP coefficients
        self.set_gains = pd.Index([])
        # Define the heat and power CHP coefficients.
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
            for j in range(0, self.lhv_table.size):
                self.gain_heat[self.lhv_table.size * i + j] = self.plant_CHP['therm_eff'][i] * \
                                                              self.lhv_table['LHV value (in J/m^3)'][j] * \
                                                              self.plant_feedstock['methane_content'][
                                                                  self.scenario_name]
                self.gain_power[self.lhv_table.size * i + j] = self.plant_CHP['elec_eff'][i] * \
                                                               self.lhv_table['LHV value (in J/m^3)'][j] * \
                                                               self.plant_feedstock['methane_content'][
                                                                   self.scenario_name]

        self.gain_heat.columns = self.set_gains
        self.gain_power.columns = self.set_gains

        # Empty control variables (are added in the inherited classes)
        self.controls = pd.Index(
            [],
            name='control_name'
        )
        # Add the chp controls (every biogas plant has at least one CHP)
        for i in range(len(self.CHP_list)):
            self.controls = pd.Index([
                # CHPs Biogas inflows
                self.plant_CHP['CHP_name'][i] + '_biogas_volume_inflow_m3_s-1'
                ]).union(self.controls)

        # State variable for storage (every bg has a storage)
        self.states = pd.Index(
            # Storage biogas content.
            self.plant_scenarios['scenario_name'] + '_storage_content_m3',
            name='state_name'
        )
        #  TODO: if the above does not work, use below
        # # State variables.
        # self.states = pd.Index(
        #     pd.concat([
        #         # Storage biogas content.
        #         self.plant_scenarios['scenario_name']
        #         + '_storage_content_m3'
        #     ]),
        #     name='state_name'
        # )

        # Output variables.
        self.outputs = pd.Index(
            # Storage biogas content.
            self.plant_scenarios['scenario_name']
            + '_storage_content_m3',
            name='output_name'
        )

        # TODO: own consumption should be here as well
        self.outputs = pd.Index([
            # net active power output
            'active_power',
            # net reactive power output
            'reactive_power',
            # net thermal output (heat)
            'thermal_power'
        ]).union(self.outputs)

        self.switches = pd.Index([])
        for i in range(len(self.CHP_list)):
            self.outputs = pd.Index([
                # CHPs active power production.
                self.plant_CHP['CHP_name'][i] + '_active_power_Wel',
                # CHPs reactive power production.
                self.plant_CHP['CHP_name'][i] + '_react_power_Var',
                # CHPs heat power production.
                self.plant_CHP['CHP_name'][i] + '_heat_Wth'
            ]).union(self.outputs)

            self.switches = pd.Index([
                # CHP switch to turn on/off
                self.plant_CHP['CHP_name'][i] + '_switch',
            ]).union(self.switches)

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

    def instantiate_state_space_matrices(self):
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

    def define_state_output_matrix(self):
        # Define the state output matrix.
        for state in self.states:
            for output in self.outputs:
                if state == output:
                    self.state_output_matrix.loc[state, output] = 1

    def define_control_output_matrix(self):
        # Define the control output matrix.
        for control in self.controls:
            for output in self.outputs:
                if ('active_power_Wel' in output) and (control[0:5] == output[0:5]):
                    self.control_output_matrix.loc[output, control] \
                        = self.gain_power[control][0] * self.plant_CHP.loc[control[0:5], 'power_factor']
                if ('react_power_Var' in output) and (control[0:5] == output[0:5]):
                    self.control_output_matrix.loc[output, control] \
                        = self.gain_power[control][0] * (1 - self.plant_CHP.loc[control[0:5], 'power_factor'])
                if ('heat_Wth' in output) and (control[0:5] == output[0:5]):
                    self.control_output_matrix.loc[output, control] \
                        = self.gain_heat[control][0]

        # add net active/reactive/thermal output
        for chp in self.plant_CHP['CHP_name'].to_list():
            for control in self.controls:
                if control[0:5] == chp:
                    self.control_output_matrix.loc['active_power', control] \
                        = self.gain_power[control][0] * self.plant_CHP.loc[control[0:5], 'power_factor']
                    self.control_output_matrix.loc['reactive_power', control] \
                        = self.gain_power[control][0] * (1 - self.plant_CHP.loc[control[0:5], 'power_factor'])
                    self.control_output_matrix.loc['thermal_power', control] \
                        = self.gain_heat[control][0]

    def define_output_constraint_timeseries(self):

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


class SimpleBiogasPlantModel(BiogasPlantModel):
    """ Creates a linear state-space model for a biogas power plant, given data about the plant.
        The class defines and creates, among other attributes, the discretized state, control, state_output and
        control_output matrices, necessary to the numerical solution of the model.
        The main difference to the FlexibleBiogasPlantModel is the lack of modelling the fermentation process (becomes constant) by:
        - Removing the biogas production rate from the variables
        - Removing the feedstock inflow from the variable
        - This biogas production rate value is used to define the disturbance_timeseries, that used to be zero. Indeed, the biogas production rate, that used to be a state/output variable, is now a disturbance
        - Disturbance_matrix is defined, while the disturbance_output_matrix remains a zero matrix
        - A major change is that the matrices are directly defined discretely, because without the digestion process discrete equations are quite obvious
        """

    model_type = 'simple'

    def __init__(
            self,
            scenario_name: str,
            timestep_start=None,
            timestep_end=None,
            timestep_interval=None,
            connect_electric_grid=True,
    ):

        super().__init__(
            scenario_name,
            timestep_start,
            timestep_end,
            timestep_interval,
            connect_electric_grid
        )

        # Get biogas production rate (constant over time)
        self.biogas_prod_rate = float(self.plant_scenarios.loc[self.scenario_name, 'init_state_digester'])

        # Disturbance variables.
        self.disturbances = pd.Index(['biogas_production_rate_m3_s-1'],
                                     name='disturbance_name')

        self.instantiate_state_space_matrices()

        # Define the matrices of the system.
        # Define the state matrix.
        self.state_matrix.loc[self.scenario_name + '_storage_content_m3',
                              self.scenario_name + '_storage_content_m3'] \
            = 1

        # Define the control matrix.
        for state in self.states:
            for control in self.controls:
                if ('storage' in state) and ('biogas' in control):
                    self.control_matrix.loc[state, control] = - self.timestep_interval.seconds

        self.define_state_output_matrix()
        self.define_control_output_matrix()

        # TODO: subtract a constant heat and power requirement

        # Define the disturbance matrix.
        self.disturbance_matrix.loc[self.scenario_name + '_storage_content_m3',
                                    'biogas_production_rate_m3_s-1'] \
            = self.timestep_interval.seconds

        self.define_initial_state()
        self.define_disturbance_timeseries()
        self.define_output_constraint_timeseries()

    # Define the initial state (either digester process starting or already active).
    def define_initial_state(self):
        self.state_vector_initial = (
            pd.Series(
                [self.plant_storage.loc[self.scenario_name, 'SOC_init_m3']],
                index=self.states
            )
        )

    # Define disturbance timeseries.
    def define_disturbance_timeseries(self):

        # Reindex, interpolate and construct full disturbance timeseries.
        self.disturbance_timeseries = pd.DataFrame(
            self.biogas_prod_rate,
            self.timesteps,
            self.disturbances
        )


class FlexibleBiogasPlantModel(BiogasPlantModel):
    """ Create a linear state-space model for a biogas power plant, given data about the plant.
    The class defines and creates, among other attributes, the discretized state, control, state_output and
    control_output matrices, necessary to the numerical solution of the model.
    """

    model_type = 'flexible'

    def __init__(
            self,
            scenario_name: str,
            timestep_start=None,
            timestep_end=None,
            timestep_interval=None,
            connect_electric_grid=True,
    ):

        super().__init__(
            scenario_name,
            timestep_start,
            timestep_end,
            timestep_interval,
            connect_electric_grid
        )

        # State variables.
        # add the biogas production state variable
        self.states = pd.Index(
            # Biogas production rate from the digester.
            self.plant_scenarios['scenario_name']
            + '_prod_biogas_m3_s-1',
            name='state_name'
        ).union(self.states)

        # Control variables.
        self.controls = pd.Index(
            # Feedstock mass flow.
            'mass_flow_kg_s-1_'
            + self.plant_scenarios['feedstock_type'],
            name='control_name'
        ).union(self.controls)

        # Output variables.
        self.outputs = pd.Index(
            # Biogas production rate from the digester / Biogas inflow to the storage.
            self.plant_scenarios['scenario_name']
            + '_prod_biogas_m3_s-1',
            name='output_name'
        ).union(self.outputs)

        self.outputs = pd.Index(
            # Active power requirement of the digester.
            self.plant_scenarios['scenario_name']
            + '_act_power_own_consumption_Wel',
            name='output_name'
        ).union(self.outputs)

        self.outputs = pd.Index(
            # Heat requirement of the digester.
            self.plant_scenarios['scenario_name']
            + '_heat_own_consumption_Wth',
            name='output_name'
        ).union(self.outputs)

        # Disturbance variables (empty).
        self.disturbances = pd.Index([None],
                                     name='disturbance_name')

        self.instantiate_state_space_matrices()

        # Heat requirement to increase the introduced feedstock's temperature
        # to the operating temperature in J/kg.
        self.gain_parasitic_heat = self.cp_water * (
                self.temp_in - self.plant_scenarios.loc[self.scenario_name, 'temperature_outside']
        )
        # Power requirement for the stirring and pumping of the feedstock in Je/kg
        # (value for cattle slurry, kept for other feedstock types).
        self.gain_parasitic_power = 25920

        # Define the matrices of the system.

        # Define the state matrix.
        self.a1 = (float(self.plant_feedstock.loc[self.scenario_name, 'time_constant_h']) * 3600) ** (-1)
        self.state_matrix.loc[self.scenario_name + '_prod_biogas_m3_s-1', self.scenario_name + '_prod_biogas_m3_s-1'] \
            = -self.a1
        self.state_matrix.loc[self.scenario_name + '_storage_content_m3', self.scenario_name + '_prod_biogas_m3_s-1'] \
            = 1.0

        # Define the control matrix.
        self.b1 = (self.plant_feedstock.loc[self.scenario_name, 'biogas_yield_m3_kgVS-1']) \
                  * self.a1 \
                  * self.plant_feedstock.loc[self.scenario_name, 'VS'] \
                  * self.plant_feedstock.loc[self.scenario_name, 'DM']
        for state in self.states:
            for control in self.controls:
                if ('prod' in state) and ('mass_flow' in control):
                    self.control_matrix.loc[state, control] = self.b1
                if ('storage' in state) and ('biogas' in control):
                    self.control_matrix.loc[state, control] = -1.0

        self.define_state_output_matrix()
        self.define_control_output_matrix()

        # subtract the own power consumption from the active power output
        self.control_output_matrix.loc['active_power',
                                       'mass_flow_kg_s-1_' + self.plant_scenarios['feedstock_type'][
                                           self.scenario_name]] \
            = - self.gain_parasitic_power
        self.control_output_matrix.loc['thermal_power',
                                       'mass_flow_kg_s-1_' + self.plant_scenarios['feedstock_type'][
                                           self.scenario_name]] \
            = - self.gain_parasitic_heat

        self.control_output_matrix.loc[
            self.scenario_name + '_act_power_own_consumption_Wel',
            'mass_flow_kg_s-1_' + self.plant_scenarios['feedstock_type'][self.scenario_name]] \
            = self.gain_parasitic_power
        self.control_output_matrix.loc[
            self.scenario_name + '_heat_own_consumption_Wth',
            'mass_flow_kg_s-1_' + self.plant_scenarios['feedstock_type'][self.scenario_name]] \
            = self.gain_parasitic_heat

        # Apply the functions.
        self.define_initial_state()
        self.define_disturbance_timeseries()
        self.define_output_constraint_timeseries()
        self.discretize_model()

    # Define the initial state (either digester process starting or already active).
    def define_initial_state(self):
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
    def define_disturbance_timeseries(self):

        # Reindex, interpolate and construct full disturbance timeseries.
        self.disturbance_timeseries = pd.DataFrame(
            0.0,
            self.timesteps,
            self.disturbances
        )

    # Define output and control constraints.
    def define_output_constraint_timeseries(self):

        super().define_output_constraint_timeseries()

        # Minimum constraint for own heat and power consumption.
        self.output_constraint_timeseries_minimum.loc[
            :, self.outputs.str.contains('_own_consumption')
        ] = 0.0

    # Discretize the model
    def discretize_model(self):
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


def make_biogas_plant_model(
    scenario_name: str,
    timestep_start=None,
    timestep_end=None,
    timestep_interval=None,
) -> BiogasPlantModel:
    """Factory method for biogas plant model based on type
    Possible types model_type:
    'simple'
    'flexible'
    """

    # Define the biogas plant model (change paths accordingly).
    base_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
    # Load the scenario.
    plant_scenarios = pd.read_csv(
        os.path.join(base_path, 'data/biogas_plant_scenario.csv')
    )
    plant_scenarios = plant_scenarios[
        plant_scenarios['scenario_name'] == scenario_name]
    plant_scenarios.index = pd.Index([scenario_name])

    # Get model type
    model_type = str(plant_scenarios.loc[scenario_name, 'model_type'])

    # Obtain biogas model classes.
    model_classes = (
        inspect.getmembers(sys.modules[__name__], lambda cls: inspect.isclass(cls) and issubclass(cls, BiogasPlantModel))
    )

    # Obtain biogas model for given `model_type`, e.g. 'simple'.
    for model_class_name, model_class in model_classes:
        if model_type == model_class.model_type:
            return model_class(scenario_name, timestep_start, timestep_end, timestep_interval)

    print(f"Can't find biogas plant model class for DER '{scenario_name}' of type '{model_type}'.")
    raise ValueError
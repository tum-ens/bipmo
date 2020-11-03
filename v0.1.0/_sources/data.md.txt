# Data Reference

``` warning::
    This reference is work in progress.
```

BiPMo is configured through the use of CSV files, where each CSV file represents a table as defined below (the file name is interpreted as the table name).

### `biogas_plant_scenario`

Defines the scenario.

| Parameter | Unit | Description |
| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier. |
| `feedstock_type` | | Feedstock identifier as defined in `biogas_plant_feedstock`. |
| `storage_name` | | Storage identifier as defined `biogas_plant_storage`. |
| `time_start` | | Start timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `time_end` | | End timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `time_step` | | Time interval in format `HH:MM:SS`. |
| `temperature_outside` | °C | Temperature outside the digester. |
| `CHP_name` | | List of the CHPs' identifiers in format `name1 name2 name3` (add the name for each new CHP included), as defined in `biogas_plant_CHP`. |
| `digester_temp` | °C | Temperature of the digestion process. |
| `init_state_digester` | m^3/s | Initial value of the biogas production rate, 0 if entry `starting`. |
| `availability_substrate_ton_per_year` | ton/a | Yearly availability of the substrate. |
| `availability_limit_type` | | Timeframe of the availability limit for the calculation of the constraint, entry `daily` or `yearly`. |
| `model_type` | | Model chosen for the simulation, with (entry `flexible`) or without (entry `simple`) the modelling of the digestion process. |
| `const_power_requirement` | W | Constant active power requirement of the biogas plant, for e.g. stirring equipment, etc. |
| `const_heat_requirement` | Wth | Constant thermal power requirement of the biogas plant to compensate heat losses of the digester. |


### `price_timeseries`

Contains price time series data.

| Parameter | Unit | Description |
| --- |:---:| --- |
| `price_type` | | Unique type identifier. |
| `time` | | Timestamp according to ISO 8601. |
| `price_value` | €/Wh | Price value. |

### `biogas_plant_CHP`

Contains data about the CHP systems used in the scenario.

| Parameter | Unit | Description |
| --- |:---:| --- |
| `CHP_name` | | Unique CHP identifier. |
| `therm_eff` | % | Thermal efficiency of the CHP. |
| `elec_eff` | % | Electrical efficiency of the CHP. |
| `elec_cap_Wel` | We | Maximum active power output of the CHP. |
| `elec_min_Wel` | We | Minimum active power output of the CHP. |
| `power_factor` | | Power factor of the generator. |
| `ramp_capacity_W_min` | We/min | Ramp capacity of the CHP. |

### `biogas_plant_feedstock`

Contains data about the feedstock used in the scenario.

| Parameter | Unit | Description |
| --- |:---:| --- |
| `feedstock_type` | | Unique feedstock identifier. |
| `DM` | % | Dry matter content of the substrate. |
| `VS` | % | Volatile solids content of the dry matter. |
| `biogas_yield_m3_kgVS-1` | m^3/s | Biogas production potential of the volatile solids contained in the substrate. |
| `methane_content` | % | Biomethane content of the biogas produced. |
| `time_constant_h` | hours | time constant for the first order digestion model. |

### `biogas_plant_storage`

Contains data about the storage used in the scenario.

| Parameter | Unit | Description |
| --- |:---:| --- |
| `storage_name` | | Unique storage identifier. |
| `SOC_min_m3` | m^3 | Minimal state of charge of the storage. |
| `SOC_max_m3` | m^3 | Maximal state of charge of the storage. |
| `SOC_init_m3` | m^3 | Initial state of charge of the storage. |
| `SOC_end` | | State of charge of the storage at the last time step, entry `init` to define the end value equal to the initial value. |
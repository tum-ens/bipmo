# Data reference

``` warning::
    This reference is work in progress.
```

BiPMo is configured through the use of CSV files, where each CSV file represents a table as defined below (the file name is interpreted as the table name).

### `biogas_plant_scenario`

Scenario definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier.|
| `electric_grid_name` | | Electric grid identifier as defined `electric grids` |
| `thermal_grid_name` | | Thermal grid identifier as defined `thermal grids` |
| `parameter_set` | | Parameter set identifier as defined in `parameters` |
| `price_type` | | Type identifier as defined in `price_timeseries` |
| `electric_grid_operation_limit_type` | | Operation limit type as defined in `electric_grid_operation_limit_types` |
| `thermal_grid_operation_limit_type` | | Type identifier as defined in `thermal_grid_operation_limit_types` |
| `timestep_start` | | Start timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_end` | | End timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_interval` | | Time interval in format `HH:MM:SS` |

### `price_timeseries`

Price time series.

| Column | Unit | Description |
| --- |:---:| --- |
| `price_type` | | Unique type identifier. |
| `time` | | Timestamp according to ISO 8601. |
| `price_value` | €/Wh | Price value |

### `biogas_plant_CHP`

| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier.|
| `electric_grid_name` | | Electric grid identifier as defined `electric grids` |
| `thermal_grid_name` | | Thermal grid identifier as defined `thermal grids` |
| `parameter_set` | | Parameter set identifier as defined in `parameters` |
| `price_type` | | Type identifier as defined in `price_timeseries` |
| `electric_grid_operation_limit_type` | | Operation limit type as defined in `electric_grid_operation_limit_types` |
| `thermal_grid_operation_limit_type` | | Type identifier as defined in `thermal_grid_operation_limit_types` |
| `timestep_start` | | Start timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_end` | | End timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_interval` | | Time interval in format `HH:MM:SS` |

### `biogas_plant_feedstock`
| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier.|
| `electric_grid_name` | | Electric grid identifier as defined `electric grids` |
| `thermal_grid_name` | | Thermal grid identifier as defined `thermal grids` |
| `parameter_set` | | Parameter set identifier as defined in `parameters` |
| `price_type` | | Type identifier as defined in `price_timeseries` |
| `electric_grid_operation_limit_type` | | Operation limit type as defined in `electric_grid_operation_limit_types` |
| `thermal_grid_operation_limit_type` | | Type identifier as defined in `thermal_grid_operation_limit_types` |
| `timestep_start` | | Start timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_end` | | End timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_interval` | | Time interval in format `HH:MM:SS` |

### `biogas_plant_storage`
| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier.|
| `electric_grid_name` | | Electric grid identifier as defined `electric grids` |
| `thermal_grid_name` | | Thermal grid identifier as defined `thermal grids` |
| `parameter_set` | | Parameter set identifier as defined in `parameters` |
| `price_type` | | Type identifier as defined in `price_timeseries` |
| `electric_grid_operation_limit_type` | | Operation limit type as defined in `electric_grid_operation_limit_types` |
| `thermal_grid_operation_limit_type` | | Type identifier as defined in `thermal_grid_operation_limit_types` |
| `timestep_start` | | Start timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_end` | | End timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_interval` | | Time interval in format `HH:MM:SS` |
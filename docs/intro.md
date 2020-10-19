# Getting Started

## Installation

### Quick installation

1. Check requirements:
    - Python 3.7
    - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository.
3. In your Python environment, run: `pip install -e path_to_repository`

### Recommended installation

The following installation procedure contains additional steps and is less prone to potential errors due to the use of Anaconda.

1. Check requirements:
    - [Anaconda Distribution](https://www.anaconda.com/distribution/) (Python 3.x version)
    - [Gurobi Optimizer](http://www.gurobi.com/) or [CPLEX Optimizer](https://www.ibm.com/analytics/cplex-optimizer)
2. Clone or download repository.
3. In Anaconda Prompt, run:
    1. `conda create -n bipmo python=3.7`
    2. `conda activate bipmo`
    3. `conda install pandas`
    4. `pip install -e path_to_repository`.


``` important::
    Please also create an issue on Github if you run into problems with the normal installation procedure.
```

## Examples

The `examples` directory contains run scripts which demonstrate possible usages of BiPMo.


## Contributing

If you are keen to contribute to this project, please see [Contributing](contributing.md).

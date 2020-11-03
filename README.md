# BiPMo
BiPMo (Biogas Plant Model) is a simulation framework for the optimal operation of a biogas plant in a given scenario that works both stand-alone and in combination with the simulation environment [FLEDGE](https://tumcreate-estl.github.io/fledge/develop/index.html).

The full documentation can be found [here](https://tum-ens.github.io/bipmo/develop/index.html)

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

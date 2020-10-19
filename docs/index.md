## FLEDGE - Flexible Distribution Grid Demonstrator

[![](https://zenodo.org/badge/201130660.svg)](https://zenodo.org/badge/latestdoi/201130660)

The Flexible Distribution Grid Demonstrator (FLEDGE) is a software tool for optimal operation problems of electric and thermal distribution grids along with distributed energy resources (DERs), such as flexible building loads, electric vehicle (EV) chargers, distributed generators (DGs) and energy storage systems (ESS). To this end, it implements 1) electric grid models, 2) thermal grid models, 3) DER models, and 4) optimal operation problems.

## Features

- Electric grid models:
    - Obtain nodal / branch admittance and incidence matrices¹.
    - Obtain steady state power flow solution for nodal voltage / branch flows / losses via fixed-point algorithm / [OpenDSS](https://github.com/dss-extensions/OpenDSSDirect.py)¹.
    - Obtain sensitivity matrices of global linear approximate grid model¹.
    - ¹Fully enabled for unbalanced / multiphase grid configuration.
- Thermal grid models:
    - Obtain nodal / branch incidence matrices.
    - Obtain thermal power flow solution for nodal head / branch flows / pumping losses.
    - Obtain sensitivity matrices of global linear approximate grid model.
- Distributed energy resource (DER) models:
    - Time series models for fixed loads.
    - Time series models for EV charging.
    - Linear models for flexible building loads.
- Optimal operation problems:
    - Obtain numerical optimization problem for combined optimal operation for electric / thermal grids with DERs via [Pyomo](https://github.com/Pyomo/pyomo).
    - Obtain electric / thermal optimal power flow solution.
    - Obtain distribution locational marginal prices (DLMPs) for the electric / thermal grids.

## Work in progress

Please note that the repository is under active development and the interface may change without notice. Create an [issue](https://github.com/TUMCREATE-ESTL/fledge/issues) if you have ideas / comments / criticism that may help to make the tool more useful.

## Contents

``` toctree::
    :maxdepth: 2

    intro
    data
    contributing
```

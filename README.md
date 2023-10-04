This repository contains all the scripts and instructions necessary to reproduce
the results presented in Garduno et al. (2024). Climate variability leads to 
multiple oxygenation episodes across the Great Oxidation Event. Submitted to GRL.

# Reproduce our results

## Install Photochempy

We made minor changes to the original Photochempy code to run our experiments. To
install the version of Photochempy that we used, clone the code with our changes
using the following command:

```
git clone -b garduno_et_al_2024 https://github.com/DanyIvan/PhotochemPy
cd PhotochemPy
```

Then create an anaconda environment to install Photochempy (using python 3.7 is
what works for me):

```
conda create -n photochempy -c conda-forge python=3.7 numpy=1.21 scipy scikit-build
conda activate photochempy
```

finally, install Photochempy with:

```
python setup.py install
```

## Run scripts

Once Photochempy is installed, you can clone this repository to run the Python scripts to reproduce our results:

```
git clone https://github.com/DanyIvan/climate_goe_over_time
```

Our code is organized in the following way:

- `input/`: Contains atmospheric profiles used in our simulations
- `output`: Folder where model output is stored
- `plots`: contains scripts and model output to reproduce plots shown in the manuscript
- `helper_funcs.py`: script with functions to define surface temperature and O2
flux evolution over time.
- `find_steady_states.py`: script to find steady states across different surface
temperatures and O2 fluxes
- `stability_analysis.py`: script to perform a stability analysis of steady states
- `run_model.py`: script to run the photochemical model over time
- `o2_flux_constant_evolution.py`: script to run the model over time with a constant
surface O2 flux
- `o2_flux_linear_increase_evolution.py`: script to run the model over time with a
linearly increasing O2 flux
- `read_output.py`: script with functions to read model output
- `qsub.py`: function to schedule a run with Sun Grid Engine queuing  system (not
necessary to reproduce results)

To rerun the experiments to create the steady state parameter space shown in
figure 1 of the manuscript run:

```
python find_steady_states.py
```

This script runs in parallel, using a different core for each surface O2 flux.

To run the simulations with a constant O2 input flux shown in figure 2 of the
manuscript, run:

```
python o2_flux_constant_evolution.py --o2_flux 1.8e12
```

for the 1.8e12 molecules/cm^2/s case, and 

```
python o2_flux_constant_evolution.py --o2_flux 2.2e12
```

for the 2.2e12 molecules/cm^2/s case.

To run the simulations with a linearly O2 input flux shown in figure 4 of the
manuscript, run:

```
python o2_flux_linear_increase_evolution.py --change_during_glaciations False
```

and

```
python o2_flux_linear_increase_evolution.py --change_during_glaciations True
```

for the case with a linear O2 flux increase and a 60% decrease during glaciations and a 60% increase after glaciations.

## Reading model output

The model output is stored in the `output/` folder. Once you run an experiment, you can use the functions in the `read_output.py` script to read the model output.

For example, to get the surface mixing ratios of a simulation:

```python
from read_output import get_surface_output
surface_output = get_surface_output('output/o2_flux_constant_1.8e12')
```

# Remake our plots

The `plots/reduced_model_output` contains a reduced version of the model output that can be used to remake our plots. To do this, go to the plots folder and run the `make_plots.py` script:

```
cd plots/
python make_plots.py
```

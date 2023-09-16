import numpy as np
from run_model import run_model, OUTPUT_FOLDER
from helper_funcs import surf_temp_evolution
import argparse

# define run parameters from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--o2_flux', default='1.8e12')
args = parser.parse_args()

o2_flux = eval(args.o2_flux)

# define surface temperature and O2 flux evolution
time = np.arange(0, 500, 0.1)
surf_temp_ev = surf_temp_evolution(time)
o2_flux_ev = np.repeat(o2_flux, len(time))

# run model
name = 'o2_flux_constant_' + args.o2_flux
run_model(name, time, o2_flux_ev, surf_temp_ev)

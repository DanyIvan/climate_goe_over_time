import numpy as np
from run_model import run_model, OUTPUT_FOLDER
from helper_funcs import surf_temp_evolution, O2_flux_evolution_linear
import argparse

# define run parameters from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--o2_flux_gl2', default='2.0e12')
parser.add_argument('--o2_flux_gl3', default='2.6e12')
parser.add_argument('--change_during_glaciations', default='False')

args = parser.parse_args()

o2_flux_gl2 = eval(args.o2_flux_gl2)
o2_flux_gl3 = eval(args.o2_flux_gl3)
change_during_glaciations = eval(args.change_during_glaciations)

# define surface temperature and O2 flux evolution
time = np.arange(0, 500, 0.1)
surf_temp_ev = surf_temp_evolution(time)
o2_flux_ev = O2_flux_evolution_linear(time, o2_flux_gl2, o2_flux_gl3)

# run model
chg_durung_gl_str = '_change_during_glaciations' if change_during_glaciations\
    else ''
name = 'linear_o2_flux_increase' + chg_durung_gl_str
run_model(name, time, o2_flux_ev, surf_temp_ev)

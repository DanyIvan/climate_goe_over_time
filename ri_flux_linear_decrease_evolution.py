import numpy as np
from run_model import run_model, OUTPUT_FOLDER
from helper_funcs import surf_temp_evolution, ri_flux_evolution_linear, glaciation_times
import argparse

# define run parameters from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--ri_flux_gl1', default='2.2e10')
parser.add_argument('--ri_flux_gl4', default='1e10')
parser.add_argument('--change_during_glaciations', default='False')

args = parser.parse_args()

ri_flux_gl1 = eval(args.ri_flux_gl1)
ri_flux_gl4 = eval(args.ri_flux_gl4)
change_during_glaciations = eval(args.change_during_glaciations)

# define surface temperature, O2 input flux, and reductant input flux evolution
time = np.arange(0, 500, 0.1)
surf_temp_ev = surf_temp_evolution(time)
o2_flux_ev = np.repeat(2e12, len(time))
if change_during_glaciations:
    for gl_time in glaciation_times:
            # decrease flux by 60% during glaiations
            idxs = np.where((time >= gl_time)&(time <= gl_time+10))[0]
            o2_flux_ev[idxs] = o2_flux_ev[idxs] - o2_flux_ev[idxs]*0.6

            # increase flux by 60% after glaiations
            idxs = np.where((time >= gl_time+10)&(time <= gl_time+20))[0]
            o2_flux_ev[idxs] = o2_flux_ev[idxs] + o2_flux_ev[idxs]*0.6

ri_flux_ev = ri_flux_evolution_linear(time, ri_flux_gl1, ri_flux_gl4)

# run model
chg_durung_gl_str = '_change_during_glaciations' if change_during_glaciations\
    else ''
name = 'linear_ri_flux_decrease' + chg_durung_gl_str
run_model(name, time, o2_flux_ev, surf_temp_ev, ri_flux_ev)

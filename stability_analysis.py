from PhotochemPy import PhotochemPy, io
from run_experiment import set_fluxes, set_atm_structure, apply_rh_profile, OUTPUT_FOLDER
import numpy as np  
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool


temp_profiles = np.loadtxt('moist_adiabat_temp_profiles.txt')
eddy_diff_profiles = np.loadtxt('moist_adiabat_eddy_profiles.txt')
surf_pressures = np.loadtxt('moist_adiabat_press_profiles.txt')[:,0]/1e5

name = 'stability_analysis'
folder = OUTPUT_FOLDER + name
Path(folder).mkdir(exist_ok=True, parents=True)

my = 365*24*60*60*1e6
def stability_analysis(o2_flux):
    '''
    Run photochemical model to steady state at given surface O2 flux 
        in a range of temperatures from 250k to 350k, and pertub the flux by
        10% 
    input:
        o2_flux (float): surface O2 flux (molecules/cm^2/s)
    '''
    pc = PhotochemPy('input/Wogan2022/species_zahnle2006.dat',
                'input/Wogan2022/reactions_new.rx',
                'input/Wogan2022/settings.yaml',
                'input/Wogan2022/atmosphere_zahnle2006_FO2=1e12_CH4O2=0.490.txt',
                'input/Wogan2022/Sun_2.4Ga.txt')

    # set atm structure
    set_atm_structure(pc, 290)
    set_fluxes(pc, o2_flux)
    apply_rh_profile(pc)

    # integreate to equilibrium at 290K
    pc.vars.equilibrium_time=my
    pc.integrate(50000,method='CVODE_BDF',atol=1e-30)
    pc.out2in()

    # save 290K steady state solution
    usol_290 = deepcopy(pc.vars.usol_out)
    
    # temperature range
    temps = np.arange(250, 360, 10)

    o2_flux_str = '{:.2e}'.format(o2_flux)
    for j, temp in enumerate(temps):
        try:
            # set atm structure for given surface temperature
            set_atm_structure(pc, temp)
            set_fluxes(pc, o2_flux)
            apply_rh_profile(pc)

            # integrate for 1my to get to steady state
            outfilename = folder + f'/{o2_flux_str}_{temp}'
            #t_eval = [1*my]
            t_eval = np.logspace(5,13.5, 500)
            last_sol, success, err = pc.photo.cvode_save(0, usol_290,t_eval,
                rtol = 1.0e-3, atol= 1e-30,use_fast_jacobian = True,
                outfilename=outfilename,
                amount2save=0)

            # perturb O2 input flux by 10%
            perturbation = o2_flux * 0.1
            set_fluxes(pc, o2_flux + perturbation)

            # integrate for 1my to get to steady state
            outfilename = folder + f'/{o2_flux_str}_{temp}_perturbed'
            usol_init = pc.vars.usol_out
            #t_eval = [0.5*my]
            t_eval = np.logspace(5,13.5, 500)
            last_sol, success, err = pc.photo.cvode_save(0, usol_init,t_eval,
                rtol = 1.0e-3, atol= 1e-30,use_fast_jacobian = True,
                outfilename=outfilename,
                amount2save=0)

        except Exception as e:
            print(Exception)
    
    
# Run each O2 flux in a different core    
o2_fluxes = np.arange(1, 6, 0.05)*1e12
with Pool() as pool:
    pool.map(stability_analysis, o2_fluxes)

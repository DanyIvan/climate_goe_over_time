from PhotochemPy import PhotochemPy
from run_model import set_o2_flux, set_atm_structure, set_rh_profile, set_ri_flux, OUTPUT_FOLDER
import numpy as np  
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool

temp_profiles = np.loadtxt('input/moist_adiabat_temp_profiles.txt')
eddy_diff_profiles = np.loadtxt('input/moist_adiabat_eddy_profiles.txt')
surf_pressures = np.loadtxt('input/moist_adiabat_press_profiles.txt')[:,0]/1e5

o2_flux = 2e12
o2_flux_str = '2e12'
def find_steady_state(ri_flux):
    '''
    Run photochemical model to steady state at a given reductant input flux
        in a range of temperatures from 250k to 350k
    input:
        ri_flux (float): surface reductant input flux (molecules/cm^2/s)
    '''
    name = f'steady_states_ri_temp_{o2_flux_str}'
    folder = OUTPUT_FOLDER + name
    Path(folder).mkdir(exist_ok=True, parents=True)
    my = 365*24*60*60*1e6
    pc = PhotochemPy('input/Wogan2022/species_zahnle2006.dat',
                'input/Wogan2022/reactions_new.rx',
                'input/Wogan2022/settings.yaml',
                'input/Wogan2022/atmosphere_zahnle2006_FO2=1e12_CH4O2=0.490.txt',
                'input/Wogan2022/Sun_2.4Ga.txt')

    # set atm structure
    set_atm_structure(pc, 290)
    set_o2_flux(pc, o2_flux)
    set_rh_profile(pc)
    set_ri_flux(pc)

    # integreate to equilibrium at 290K
    pc.vars.equilibrium_time=my
    pc.integrate(50000,method='CVODE_BDF',atol=1e-30)
    pc.out2in()

    # save 290K steady state solution
    usol_290 = deepcopy(pc.vars.usol_out)
    
    # temperature range
    temps = np.arange(250, 360, 10)
    
    time = np.linspace(0, 1, 10) * my
    t0 = time[0]
    usol_init = pc.vars.usol_init
    ri_flux_str = '{:.2e}'.format(ri_flux)
    for temp in temps:
        # find steady state starting from 290K
        usol_init = usol_290
        for i in range(len(time) -1):
            t_eval = [time[i+1]]

            # set atm structure for given surface temperature
            set_atm_structure(pc, temp)
            set_o2_flux(pc, o2_flux)
            set_rh_profile(pc)
            set_ri_flux(pc, ri_flux=ri_flux)

            # run the model
            t0 = time[i]
            outfilename = folder + f'/{ri_flux_str}_{temp}_{i}'
            last_sol, success, err = pc.photo.cvode_save(t0, usol_init,t_eval,
                rtol = 1.0e-3, atol= 1e-30,use_fast_jacobian = True,
                outfilename=outfilename,
                amount2save=0)
            if success:
                last_success_sol = last_sol

            # use last succesful solution and inital condition
            usol_init = last_success_sol



# Run each ri flux in a different core
ri_fluxes =  np.linspace(3.2e10, 0.5e9, 35)
with Pool() as pool:
    pool.map(find_steady_state, ri_fluxes)
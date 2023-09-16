from PhotochemPy import PhotochemPy, io
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_FOLDER = 'output/'

temp_profiles = np.loadtxt('input/moist_adiabat_temp_profiles.txt')
eddy_diff_profiles = np.loadtxt('input/moist_adiabat_eddy_profiles.txt')
surf_press = np.loadtxt('input/moist_adiabat_press_profiles.txt')[:,0]/1e5

def interp_profiles(profiles, surf_temp):
    '''Interpolate temperature or eddy diffusivity profile given the surface 
    temperature   
    input:
        profiles (np.array 1d): array of known profiles
        surf_temp (float): surface temperature of interpolated profile (k)
    returns:
        interpolated_profile (np.array)
    '''
    temps = np.arange(240, 360.5, 0.5)
    nz = profiles.shape[1]
    interpolated_profile = np.zeros(nz)
    for i in range(0, nz):
        level = profiles[:, i]
        interpolated_profile[i] = np.interp(surf_temp, temps, level)
    return interpolated_profile

def interp_press(surf_temp):
    '''
    Interpolate surface pressure (bars) given the surface temperature (k)
    '''
    temps = np.arange(240, 360.5, 0.5)
    return np.interp(surf_temp, temps, surf_press)

def set_fluxes(pc, o2_flux):
    '''
    Sets O2 and CH4 surface input fluxes to photochemical model with a 0.49 ratio
    input:
        pc (photochempy object): instance of photochempy class
        o2_flux (float): O2 surface flux (molecules/cm^2/s)
    returns:
        none
    '''
    pc.set_lbound('O2',2)
    pc.set_lbound('CH4',2)
    pc.set_surfflux('O2', o2_flux)
    pc.set_surfflux('CH4', o2_flux * 0.49)
    

def set_atm_structure(pc, surf_temp):
    '''
    Sets temperature, pressure and eddy diffusivity profiles, and tropopause
    height in photochemical model 
    input:
        pc (photochempy object): instance of photochempy class
        surf_temp (float): surface temperature (k)
    returns:
        none
    '''
    temp_profile = interp_profiles(temp_profiles, surf_temp)
    edd_profile = interp_profiles(eddy_diff_profiles, surf_temp)
    p0 = interp_press(surf_temp)
    pc.vars.t[:] = temp_profile
    pc.vars.edd[:] = edd_profile
    pc.data.p0 = p0

    jtrop = np.argmin(temp_profile) + 1
    pc.data.jtrop = jtrop
    pc.data.ztrop = pc.data.z[jtrop]

def set_rh_profile(pc, rh=0.35):
    '''
    Sets relative humidity profile in photochemical model 
    input:
        pc (photochempy object): instance of photochempy class
        rh (float): relative humidity
    returns:
        none
    '''
    pc.data.use_manabe = False
    pc.data.relative_humidity = rh


def run_model(folder_name, time, o2_flux_ev, surf_temp_ev):
    '''
    Run experiment for a given O2 surface flux and temperature evolution
    input:
        folder_name (str): name of folder to store output
        time (1d array): array with times (my) when temperature and the O2 surface 
            flux will be updated
        o2_flux_ev (1d array): array of O2 surface fluxes at times defined in time
        surf_temp_ev (1d array): array of surface temperatures at times defined 
            in time
    returns:
        none
    '''
    my = 365*24*60*60*1e6
    pc = PhotochemPy('input/Wogan2022/species_zahnle2006.dat', \
        'input/Wogan2022/reactions_new.rx', \
        'input/Wogan2022/settings.yaml', \
        'input/Wogan2022/atmosphere_zahnle2006_FO2=1e12_CH4O2=0.490.txt',
        'input/Wogan2022/Sun_2.4Ga.txt')

    # set initial conditions
    set_fluxes(pc, o2_flux_ev[0])
    set_atm_structure(pc, surf_temp_ev[0])
    set_rh_profile(pc)

    # integrate to equilibrium
    pc.vars.equilibrium_time=1e13
    pc.integrate(50000,method='CVODE_BDF',atol=1e-30)
    pc.out2in()

    # make output folder and save surface o2 flux evolution
    folder = OUTPUT_FOLDER + folder_name
    Path(folder).mkdir(exist_ok=True, parents=True)
    np.savetxt(f'{folder}/o2_flux.txt', o2_flux_ev)

    # evolve over time
    t0 = time[0]
    usol_init = pc.vars.usol_init
    for i in range(len(time) -1):
        t_eval = [time[i+1] * my]

        # set atm conditions
        set_fluxes(pc, o2_flux_ev[i])
        set_atm_structure(pc, surf_temp_ev[i])
        set_rh_profile(pc)

        # integrate
        t0 = time[i] * my
        last_sol, success, err = pc.photo.cvode_save(
            t0,
            usol_init,
            t_eval,
            rtol = 1.0e-3,
            atol= 1e-30,
            use_fast_jacobian = True,
            outfilename=f'{folder}/{i}',
            amount2save=0)
        if success:
            last_success_sol = last_sol

        # use last succesful solution and inital condition
        usol_init = last_success_sol

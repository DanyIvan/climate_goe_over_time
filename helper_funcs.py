import numpy as np

# define glaciation times
glc_t1 = 2430
glc_t2 = 2375
glc_t3 = 2330
glc_t4 = 2250
glaciation_times = [glc_t1, glc_t2, glc_t3, glc_t4]
glaciation_times = [2500 - x for x in glaciation_times]

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

def surf_temp_evolution(time, T_tmpt=290, T_mg=320, T_gl=250):
    '''
    Defines surface temperature evolution over time
    input:
        time (1d array): time in my
        T_tmpt (float): surface temperature (k) during temperate climates
        T_mg (float): surface temperature (k) during moist greenhouse climates
        T_gl (float): surface temperature (k) during glacial climates
    returns
        surf_temp_ev (1d array): surface temperature evolution over time
    '''
    def surf_temp(time):
        if time < glaciation_times[0]:
            return T_tmpt
        elif time >= glaciation_times[0] and time < glaciation_times[0]+10:
            return T_gl
        elif time >= glaciation_times[0]+10 and time < glaciation_times[1]:
            return sigmoid(time, L=T_tmpt-T_mg, x0=glaciation_times[0] + 15, k=.75, b=T_mg)
        elif time >= glaciation_times[1] and time < glaciation_times[1]+10:
            return T_gl
        elif time >= glaciation_times[1]+10 and time < glaciation_times[2]:
            return sigmoid(time, L=T_tmpt-T_mg, x0=glaciation_times[1]+15, k=.75, b=T_mg)
        elif time >= glaciation_times[2] and time < glaciation_times[2]+10:
            return T_gl
        elif time >= glaciation_times[2]+10 and time < glaciation_times[3]:
            return sigmoid(time, L=T_tmpt-T_mg, x0=glaciation_times[2]+15, k=.75, b=T_mg)
        elif time >= glaciation_times[3] and time < glaciation_times[3] + 10:
            return T_gl
        elif time >= glaciation_times[3] + 10 and time < 300:
            return sigmoid(time, L=T_tmpt-T_mg, x0=glaciation_times[3]+15, k=.75, b=T_mg)
        elif time >= 300:
            return T_tmpt 
    return np.array(list(map(surf_temp, time)))
        

def O2_flux_evolution_linear(time, o2_flux_0, o2_flux_gl3, 
        change_during_glaciations=False):
    '''
    Defines O2 flux linear increase evolution over time.
    input:
        time (1d array):  time in my
        o2_flux_0 (float): O2 flux (molecules/cm^2/s) at beginning of run
        o2_flux_gl3 (float): O2 flux (molecules/cm^2/s) during 3rd glaciation
        change_during_glaciations (bool): decrease O2 flux by 60% during 
            glaciations and increase it by 60% during moist greenhouse climates
            if true
    returns:
        o2_flux_ev (1d array): o2 surface flux evolution over time
    '''
    times = (0, glaciation_times[2] + 10)
    o2_fluxes = (o2_flux_0, o2_flux_gl3)
    a,b = np.polyfit(times, o2_fluxes, deg=1)
    o2_flux_ev = a * time + b
    
    if change_during_glaciations:
        for gl_time in glaciation_times:
            # decrease flux by 60% during glaiations
            idxs = np.where((time >= gl_time)&(time <= gl_time+10))[0]
            o2_flux_ev[idxs] = o2_flux_ev[idxs] - o2_flux_ev[idxs]*0.6

            # increase flux by 60% after glaiations
            idxs = np.where((time >= gl_time+10)&(time <= gl_time+20))[0]
            o2_flux_ev[idxs] = o2_flux_ev[idxs] + o2_flux_ev[idxs]*0.6

    return o2_flux_ev

def ri_flux_evolution_linear(time, ri_flux_0, ri_flux_gl3):
    '''
    Defines reductant input flux linear decrease evolution over time.
    input:
        time (1d array):  time in my
        ri_flux_0 (float): ri flux (molecules/cm^2/s) at beginning of run
        ri_flux_gl3 (float): ri flux (molecules/cm^2/s) during 3rd glaciation
    returns:
        ri_flux_ev (1d array): ri surface flux evolution over time
    '''
    times = (0, glaciation_times[3] + 10)
    # times = (0, 500)
    ri_fluxes = (ri_flux_0, ri_flux_gl3)
    a,b = np.polyfit(times, ri_fluxes, deg=1)
    ri_flux_ev = a * time + b
    
    ctime = glaciation_times[3] + 10
    ri_flux_ev[ctime:] = ri_flux_ev[ctime]
    return ri_flux_ev

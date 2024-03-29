from PhotochemPy import io
import pandas as pd
import numpy as np
import os

def get_steady_state_output(outfolder, flux='O2', filename_suffix='_8'):
   '''
   Reads steady states surface output stored in outfolder
   input:
      outfolder (str): folder path where output is stored
      flux (str): if equals to "O2" the function will read files from an experiment 
         in which O2 flux was varied to find steady states. If equals to "ri"
         the function will read files from an experiment in which the reductant 
         input flux was varied to find steady states
      filename_suffix (str): suffix of the files to read
   returns:
      steady_states (pandas dataframe): dataframe with surface output
   '''

   # define surface O2 flux and temp range for steady states
   O2_fluxes = np.arange(1, 6, 0.05)*1e12
   ri_fluxes =  np.linspace(3.3326e10, 1e9, 30)
   temps = np.arange(250, 360, 10)

   sol_dict = {
      'time': [],
      'O2': [],
      'OH': [],
      'H2O':[],
      'CH4': [],
      'O3': [],
      'S8AER': [],
      'T_time': [],
      'den': []
               }

   fluxes_included = []
   if flux == 'O2':
      fluxes = O2_fluxes
      dict_flux_str = 'O2_flux'
   elif flux == 'ri':
      fluxes = ri_fluxes
      dict_flux_str = 'ri_flux'

   for flux in fluxes:
      flux_str = '{:.2e}'.format(flux)
      for temp in temps:
         try:
            # read last solution
            filename = f'{outfolder}/{flux_str}_{temp}' + filename_suffix
            sol = io.read_evolve_output(filename)
            # ignore solutions where the model broke (they have O2 surface values
            # lower than 1e12)
            if sol['O2'][0][0] > 1E-12:
               fluxes_included.append(flux)
               for key in sol_dict.keys(): 
                  sol_dict[key].append(sol[key][0])         
         except Exception as e:
               print(e)
    
   for key in sol_dict.keys():
      if key == 'time':
         sol_dict[key] = np.hstack([sol_dict[key]])
      else:
         sol_dict[key] = np.vstack(sol_dict[key])

   # create dataframe with surface output
   surface_sol_dict = {
      'time': sol_dict['time'],
      'O2': sol_dict['O2'][:, 0],
      'CH4': sol_dict['CH4'][:, 0],
      'OH': sol_dict['OH'][:, 0],
      'H2O': sol_dict['H2O'][:, 0],
      'O3_col': np.trapz(sol_dict['O3'] * sol_dict['den'],
               np.arange(0.25, 100, 0.5)*1e5, axis=1),
      'S8AER_col': np.trapz(sol_dict['S8AER'] * sol_dict['den'],
               np.arange(0.25, 100, 0.5)*1e5, axis=1),
      'T_time': sol_dict['T_time'][:, 0],
      dict_flux_str: fluxes_included
   }

   steady_states = pd.DataFrame(surface_sol_dict)
   return steady_states

def get_stability_analysis_output(outfolder):
   '''
   Reads steady states surface output from stability analysis output stored in
      outfolder
   input:
      outfolder (str): folder path where output is stored
   returns:
      steady_states (pandas dataframe): dataframe with stability analysis
         surface output
   '''
   output = get_steady_state_output(outfolder, filename_suffix='')
   output['type'] = 'steady state'
   output_perturbed = get_steady_state_output(outfolder,
      filename_suffix='_perturbed')
   output_perturbed['type'] = '5pct perturbed steady state'
   output = pd.concat([output, output_perturbed])
   return output
   

def get_surface_output(outfolder):
   '''
   Reads surface output from experiment stored in outfolder
   input:
      outfolder (str): folder path where output is stored
   returns:
      surface_output (pandas dataframe): dataframe with surface output
   '''
   files = [f for f in os.listdir(outfolder) if not f.endswith('.txt')]
   Nfiles = len(files)

   # read O2 flux
   o2_fluxes = np.loadtxt(f'{outfolder}/o2_flux.txt')
   ri_fluxes = np.loadtxt(f'{outfolder}/ri_flux.txt')

   sol_dict = {
      'time': [],
      'O2': [],
      'OH': [],
      'H2O':[],
      'CH4': [],
      'O3': [],
      'S8AER': [],
      'T_time': [],
      'den': [],
      'h2osat':[]
      }

   sol_idxs = []
   for i in range(Nfiles-1):
       try:
           # read solution
           sol = io.read_evolve_output(f'{outfolder}/{i}')
           # ignore runs in which model broke
           if sol['O2'][0][0] > 1E-12:
               sol_idxs.append(i)
               for key in sol_dict.keys():
                   sol_dict[key].append(sol[key][0])
       except Exception as e:
           print(e)

   for key in sol_dict.keys():
      if key == 'time' or key == 'O2_flux':
         sol_dict[key] = np.hstack([sol_dict[key]])
      else:
         sol_dict[key] = np.vstack(sol_dict[key])

   # create dataframe with surface output
   surface_sol_dict = {
         'time': sol_dict['time'],
         'O2': sol_dict['O2'][:, 0],
         'CH4': sol_dict['CH4'][:, 0],
         'OH': sol_dict['OH'][:, 0],
         'H2O': sol_dict['H2O'][:, 0],
         'O3_col': np.trapz(sol_dict['O3'] * sol_dict['den'],
                  np.arange(0.25, 100, 0.5)*1e5, axis=1),
         'S8AER_col': np.trapz(sol_dict['S8AER'] * sol_dict['den'],
                  np.arange(0.25, 100, 0.5)*1e5, axis=1),
         'T_time': sol_dict['T_time'][:, 0],
         'O2_flux': [o2_fluxes[i] for i in sol_idxs],
         'ri_flux': [ri_fluxes[i] for i in sol_idxs]
      }

   surface_output = pd.DataFrame(surface_sol_dict)
   return surface_output


def get_profiles(outfolder):
   '''
   Reads profiles from experiment stored in outfolder
   input:
      outfolder (str): folder path where output is stored
   returns:
      profiles (pandas dataframe): dataframe with species profiles
   '''
   files = [f for f in os.listdir(outfolder) if not f.endswith('.txt')]
   Nfiles = len(files)
   print(Nfiles)

   # read O2 fluxes
   o2_fluxes = np.loadtxt(f'{outfolder}/o2_flux.txt')
   ri_fluxes = np.loadtxt(f'{outfolder}/ri_flux.txt')

   # read profiles and store them in a dataframe
   data = []
   for i in range(Nfiles-1):
      sol = io.read_evolve_output(f'{outfolder}/{i}')
         # ignore runs in which model broke
      if sol['O2'][0][0] > 1e-12:
            sp_sol_dict = {
               'O2': sol['O2'][0] ,
               'CH4': sol['CH4'][0],
               'O3': sol['O3'][0],
               'H2O': sol['H2O'][0],
               'OH': sol['OH'][0],
               'H2': sol['H2'][0],
               'CO': sol['CO'][0],
               'T_time': sol['T_time'][0],
               'alt': sol['alt'],
               'rh': sol['H2O'][0]/sol['h2osat'][0]

            }
            sol_df = pd.DataFrame(sp_sol_dict)
            sol_df['time'] = sol['time'][0]
            sol_df['O2_flux'] = o2_fluxes[i]
            sol_df['ri_flux'] = ri_fluxes[i]
            data.append(sol_df)

   data = pd.concat(data)
   return data
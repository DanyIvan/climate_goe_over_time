import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import math as m
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerBase
import copy

# size of figure to fit in a4 page size
a4_x, a4_y = (8.27, 11.69)

# set ticks size
matplotlib.rc('xtick', labelsize=6) 
matplotlib.rc('ytick', labelsize=6)

def add_colorbar(fig, axes, values, label, palette='coolwarm', fraction=0.05,
    pad=0.08, aspect=70, fontsize=8, ticks=[], lognorm=False):
    '''Adds a colorbar to a figure'''
    cmap =  sns.color_palette(palette, len(values), as_cmap=True)
    if lognorm:
        norm = matplotlib.colors.LogNorm(vmin=min(values), vmax=max(values))
    else:
        norm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=axes, orientation='horizontal', fraction=fraction, pad=pad,
        aspect=aspect)
    cb.set_label(label, fontsize=fontsize)
    if len(ticks) != 0:
        cb.set_ticks(ticks)
    return cb

def sci_notation(num, decimal_digits=2):
    '''Converts a number to a string in scientific notation'''
    exponent = int(m.floor(m.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    precision = decimal_digits
    return r"${}\times 10^{{{}}}$".format(coeff, exponent)

def subplots_centered(nrows, ncols, figsize, nfigs):
    """
    Source: https://stackoverflow.com/questions/53361373/center-the-third-
        subplot-in-the-middle-of-second-row-python
    Modification of matplotlib plt.subplots(),
    useful when some subplots are empty.
    
    It returns a grid where the plots
    in the **last** row are centered.
    
    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
    """
    assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
    
    fig = plt.figure(figsize=figsize)
    axs = []
    
    m = nfigs % ncols
    m = range(1, ncols+1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m*ncols)

    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows-1: # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
        axs.append(ax)
        
    return fig, axs

def parameter_space():
    '''
    Plots parameter space of species evolution at prescribed surface temperatures
    and O2 input fluxes (figure 1 in manuscript).
    '''
    # read model output
    output_stability = pd.read_csv('reduced_model_output/stability_analysis.csv')
    output = output_stability[output_stability.type == 'steady state']
    output_pert = output_stability[
        output_stability.type == '5pct perturbed steady state']
    
    # get points that change more than a factor of 2 as a result of 5% O2
    # flux perturbation
    stability_df = output[['T_time', 'O2_flux', 'O2', 'O3_col', 'CH4',
        'S8AER_col']]
    # perturbed O2
    output_pert = output_pert[['O2', 'T_time', 'O2_flux']].\
        rename(columns={'O2':'O2_final'})
    stability_df = pd.merge(stability_df, output_pert, on=['T_time', 'O2_flux'])
    # multiply O2 by 2
    stability_df['O2_doubled'] = stability_df['O2'] * 2
    # if perturbed > O2_half_om ==> NOT STABLE
    stability_df['stable'] =\
        stability_df['O2_final'] < stability_df['O2_doubled']
    unstable_ponts = stability_df[stability_df.stable==False]
    
    # highlighted fluxes
    specific_fluxes = [1.2e12, 2e12, 2.8e12]
    linestyles = ['solid', 'dotted', 'dashed']

    # colors of O2 fluxes
    O2_fluxes = output.O2_flux.unique()
    colors = sns.color_palette('viridis', len(O2_fluxes))
    
    labels = [r'$O_2$', r'$O_3$', r'$CH_4$', r'$S_8$']
    fig, axs = plt.subplots(2,2, figsize=(a4_x * 0.7, a4_y * 0.4))
    axs = axs.flatten()
    for j,sp in enumerate(['O2', 'O3_col', 'CH4', 'S8AER_col']):
        # plot species
        plt.sca(axs[j])
        for i, flux in enumerate(O2_fluxes):
            d = output[output.O2_flux == flux]
            plt.plot(d.T_time, d[sp], color=colors[i], lw = 0.8, alpha=0.7)
        
        plt.scatter(unstable_ponts.T_time, unstable_ponts[sp],
            s=6, color='k',marker='x', linewidths=0.5, zorder=5)
        # highlight specific fluxes
        for i, flux in enumerate(specific_fluxes):
            d = output[round(output.O2_flux) == flux]
            plt.plot(d.T_time, d[sp], color='#F68109', ls=linestyles[i], lw=0.8,
                    label=sci_notation(flux))
        if j in [2, 3]:
            plt.xlabel('Surface temperature (K)', fontsize=7)
        if j in [0,2]:
            plt.ylabel(f'{labels[j]} surface mixing ratio', fontsize=7)
        else:
            plt.ylabel(f'{labels[j]} column'+ r' ($/cm^2$)', fontsize=7)
        if j != 3:    
            plt.yscale('log')
        if j == 3:
            plt.legend(fontsize=5, title=r'$O_2$ flux ($/cm^2/s$)',
                    title_fontsize=6, frameon=False)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    cb = add_colorbar(fig, axs, O2_fluxes, r'$O_2$ flux ($/cm^2/s$)', palette='viridis',
            fraction=0.1, pad=0.12, lognorm=True, aspect=50)
    for i, flux in enumerate(specific_fluxes):
        cb.ax.axvline(x=flux, lw=0.9, c='#F68109', linestyle=linestyles[i])

    plt.savefig('parameter_space.pdf', bbox_inches='tight')

def results_with_proxies(o2_flux_increase=False, ri_flux_decrease=False,
        profiles=False):
    '''
    Plots evolution of species with time along with proxies (figures 2 and 3 
    in manuscript)
    '''
     # read proxies file
    proxies = pd.read_csv('goe_O2_proxies.csv')

    # read output
    if o2_flux_increase:
        output = pd.read_csv('reduced_model_output/o2_flux_linear_increase.csv')
    elif ri_flux_decrease:
        output = pd.read_csv('reduced_model_output/ri_flux_linear_decrease.csv')
    else:
        output = pd.read_csv('reduced_model_output/o2_flux_constant.csv')
    output = output[output.O2 > 1e-12]

    # change time scale
    my = 365*24*60*60*1e6
    output['time'] = np.abs(output['time']/my - 2500)

    if o2_flux_increase or ri_flux_decrease:
        output_case1 = output[output.change_during_glaciations]
        output_case2 = output[output.change_during_glaciations == False] 
        # do not consider point at which model broke
        output_case1 = output_case1[output_case1.time != 2115.8]    
    else:
        output_case1 = output[output.O2_flux == 1.8e12]
        output_case2 = output[output.O2_flux == 2.2e12]
        
    # define glaciation times (Gumsley et al 2017)
    glc_t1 = 2430
    glc_t2 = 2375
    glc_t3 = 2330
    glc_t4 = 2250
    glaciation_times = [glc_t1, glc_t2, glc_t3, glc_t4]

    if profiles:
        fig, axs = plt.subplot_mosaic(
            '11111;22222;33333;44444;55555;66666;77777;88888;abcde',
            figsize=(a4_x * 0.75, a4_y * 0.65),
            gridspec_kw={'height_ratios': [1, 0.5, 1, 1, 1, 1, 1, 0.5, 1.8]})
        axs = list(axs.values())
    else:
        fig, axs = plt.subplots(7, 1, figsize=(a4_x * 0.75, a4_y * 0.55),
            gridspec_kw={'height_ratios': [1, 0.5, 1, 1, 1, 1, 1]})
    
    # panel A: Temperature 
    # --------------------------------------------------------------------------
    plt.sca(axs[0])
    axs[0].invert_xaxis()
    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.xlim([2520, 1980])
    plt.plot(output_case1.time, output_case1.T_time, color='k', lw=1)
    plt.ylabel('Temperature (k)', fontsize='6')
    axs[0].text(0.01, 0.9, 'a)', transform=axs[0].transAxes, size=6)
    plt.ylim([235, 335])

    # plot times where profiles will be made
    profile_points_age = [2460, 2425, 2420]
    profile_points_temp = [290, 250, 320]
    if not (o2_flux_increase or ri_flux_decrease):
        plt.scatter(profile_points_age, profile_points_temp, marker='x',
                    color='k', s=14, zorder=10, linewidths=0.8)

    # plot temperature constraints and glaciations
    for gl_time in glaciation_times:
        axs[0].axvspan(gl_time, gl_time-10, alpha=0.6, color='#6495ED')
        axs[0].arrow(gl_time - 5, 248, 0, -6, head_width=4, 
            width=0, head_length=4, fc='#6495ED', ec='#6495ED')
        axs[0].hlines(y=248, xmin=gl_time + -10, xmax=gl_time, linewidth=0.8,
            color='#6495ED')
        axs[0].arrow(gl_time - 15, 320, 0, 6, head_width=4, 
            width=0, head_length=4, fc='#de626d', ec='#de626d')
        axs[0].hlines(y=321, xmin=gl_time -11, xmax=gl_time -19, linewidth=0.8,
            color='#de626d')
        
    # legend
    axs[0].arrow(2100, 310, 0, -6, head_width=4, 
        width=0, head_length=4, fc='#6495ED', ec='#6495ED')
    axs[0].hlines(y=310, xmin=2100-4, xmax=2100+4, linewidth=0.8,
        color='#6495ED')
    plt.text(2100-10, 298, 'Global glaciation\ntemperature constraint',
        fontsize=4)
    axs[0].arrow(2100, 320, 0, 6, head_width=4, 
        width=0, head_length=4, fc='#de626d', ec='#de626d')
    axs[0].hlines(y=320, xmin=2100-4, xmax=2100+4, linewidth=0.8,
        color='#de626d')
    plt.text(2100-10, 317, 'Hot-moist greenhouse\ntemperature constraint',
        fontsize=4)

        
    # panel A: evaporite record
    # ---------------------------
    plt.sca(axs[1])
    axs[1].invert_xaxis()
    plt.setp(axs[1].get_xticklabels(), visible=False)
    plt.xlim([2520, 1980])
    # plt glaciations
    for gl_time in glaciation_times:
        axs[1].axvspan(gl_time, gl_time-10, alpha=0.6, color='#6495ED')
    
    # plot evaporite record (Bekker & Holland, 2012)
    width = 0.15 #if not profiles else 0.1
    se1 = matplotlib.patches.Rectangle((glc_t3 - 12, 0.65), -6, width, color='#de626d')
    se2 = matplotlib.patches.Rectangle((glc_t4 - 20, 0.65), -150, width, color='#de626d')
    mq1 = matplotlib.patches.Rectangle((glc_t3 - 12,0.25), -6, width, color='#de626d')
    mq2 = matplotlib.patches.Rectangle((glc_t4 - 20,0.25), -6, width, color='#de626d')
    for i in range(6):
        se = matplotlib.patches.Rectangle((2075 -12*i, 0.65), -6, width, color='#de626d')
        axs[1].add_patch(se)
    axs[1].add_patch(se1)
    axs[1].add_patch(se2)
    axs[1].add_patch(mq1)
    axs[1].add_patch(mq2)
    plt.text(glc_t3 - 12, 0.08, 'Mature quartz sandstones', fontsize=5)
    plt.text(glc_t3 - 12, 0.45, 'Marine sulfate evaporites', fontsize=5)
    axs[1].axis('off')
     
    # panel B: O2 input flux and reductant input flux
    # -------------------------------------------------------------------------
    plt.sca(axs[2])
    axs[2].invert_xaxis()
    plt.setp(axs[2].get_xticklabels(), visible=False)
    plt.xlim([2520, 1980])
    for gl_time in glaciation_times:
        axs[2].axvspan(gl_time, gl_time-10, alpha=0.6, color='#6495ED')
    plt.plot(output_case1.time, output_case1.O2_flux, color='k', lw=1, 
        label=r'$O_2$ surface flux')
    plt.plot(output_case2.time, output_case2.O2_flux, color='k', lw=1, ls=':')
    plt.ylabel(r'$O_2$ surface' + '\n' + r'flux ($/cm^2/s$)', fontsize='6')
    if not (o2_flux_increase or ri_flux_decrease):
        plt.ylim([1.6e12, 2.4e12])
    axs[2].text(0.01, 0.9, 'b)', transform=axs[2].transAxes, size=6)
    ax2 = axs[2].twinx()
    ax2.plot(output_case2.time, output_case2.ri_flux, color='#a56daa', lw=1,
        label=r'Ri surface flux')
    ax2.set_ylabel(r'Ri surface' + '\n' + r'flux ($/cm^2/s$)', fontsize='6',
        color='#a56daa')
    if not ri_flux_decrease:
        axs[2].set_zorder(ax2.get_zorder()+10)
        axs[2].set_frame_on(False)
    ax2.tick_params(axis='y', labelcolor='#a56daa')


    # panel C: O2  mixing ratio
    # -------------------------------------------------------------------------
    plt.sca(axs[3])
    axs[3].invert_xaxis()
    plt.setp(axs[3].get_xticklabels(), visible=False)
    plt.xlim([2520, 1980])
    plt.plot(output_case1.time, output_case1.O2, color='k', lw=1, label=r'$O_2$')
    plt.plot(output_case2.time, output_case2.O2, color='k', lw=1, ls=':', label=r'$O_2$')
    plt.yscale('log')
    plt.ylim([1e-9, 1e-2])
    axs[3].text(0.01, 0.9, 'c)', transform=axs[3].transAxes, size=6)
    plt.ylabel(r'$O_2$ mixing ratio', fontsize='6')
    for gl_time in glaciation_times:
        axs[3].axvspan(gl_time, gl_time-10, alpha=0.6, color='#6495ED')
    
    # plot mif constraints. I choose ages based on nearness to glaciations
    transvaal_mif_times =  proxies[(proxies.Basin=='Transvaal')&\
        (proxies.Proxy=='MIF-S')]['Assigned date Mya']
    for time in transvaal_mif_times:
        axs[3].arrow(time, 1e-6, 0, -5e-7, head_width=4, 
            width=0, head_length=1.2e-7, fc='k', ec='k')
        axs[3].hlines(y=1e-6, xmin=time - 5, xmax=time + 5, linewidth=0.8,
            color='k')
        # if time in [2284, 2247]:
        #     plt.text(time - 5, 4e-7, '?', fontsize=7)
    
    transvaal_mdf_times = proxies[(proxies.Basin=='Transvaal')&\
        (proxies.Proxy=='No MIF-S')]['Assigned date Mya']
    for time in transvaal_mdf_times:
        axs[3].arrow(time, 1e-5, 0, 9e-6, head_width=4, 
            width=0, head_length=1.2e-5, fc='#ad941f', ec='#ad941f')
        axs[3].hlines(y=1e-5, xmin=time - 5, xmax=time + 5, linewidth=0.8,
            color='#ad941f')
        
    # # legend
    if not (o2_flux_increase or ri_flux_decrease):
        axs[3].arrow(2100, 2e-5, 0, -1.3e-5, head_width=4, 
            width=0, head_length=0.2e-5, fc='k', ec='k')
        axs[3].hlines(y=2e-5, xmin=2100 - 5, xmax=2100 + 5, linewidth=0.8,
            color='k')
        plt.text(2100 -10, 0.5e-5, 'MIF-S constraints', fontsize=5)
        axs[3].arrow(2100, 1e-4, 0, 1e-4, head_width=4, 
            width=0, head_length=1.1e-4, fc='#ad941f', ec='#ad941f')
        axs[3].hlines(y=1e-4, xmin=2100 - 5, xmax=2100 + 5, linewidth=0.8,
            color='#ad941f')
        plt.text(2100 -10, 1.3e-4, 'No MIF-S constraints', fontsize=5)
    else:
        axs[3].arrow(2100, 1e-7, 0, -6e-8, head_width=4, 
            width=0, head_length=1.2e-8, fc='k', ec='k')
        axs[3].hlines(y=1e-7, xmin=2100 - 5, xmax=2100 + 5, linewidth=0.8,
            color='k')
        plt.text(2100 -10, 5e-7, 'No MIF-S constraints', fontsize=5)
        axs[3].arrow(2100, 3e-7, 0, 4e-7, head_width=4, 
            width=0, head_length=5e-7, fc='#ad941f', ec='#ad941f')
        axs[3].hlines(y=3e-7, xmin=2100 - 5, xmax=2100 + 5, linewidth=0.8,
            color='#ad941f')
        plt.text(2100 -10, 3.5e-8, 'MIF-S constraints', fontsize=5)
    
     # Panel C: proxy records
    # -------------------------------------------------------------------
    plt.sca(axs[4])
    axs[4].invert_xaxis()
    plt.setp(axs[4].get_xticklabels(), visible=False)
    plt.xlim([2520, 1980])
    plt.ylim([0,1])
    # plot formation rectangles
    transvaal_boxes = [
        {'start': 2500, 'span': glc_t1 - 2500 + 3, 'fc':(1,1,1,0)},
        {'start': glc_t1, 'span': -10, 'fc':'#6495ED'},
        {'start': glc_t1, 'span': glc_t2 - glc_t1 + 3, 'fc':(1,1,1,0)},
        {'start': glc_t2, 'span': -10, 'fc':'#6495ED'},
        {'start': glc_t2, 'span': glc_t3 - glc_t2 + 3, 'fc':(1,1,1,0)},
        {'start': glc_t3, 'span': -10, 'fc':'#6495ED'},
        {'start': glc_t3, 'span': -76, 'fc':(1,1,1,0)},
        {'start': glc_t4, 'span': -10, 'fc':'#6495ED'},
        {'start': glc_t4, 'span': -50, 'fc':(1,1,1,0)}
    ]

    huronian_boxes = [
        {'start': 2500, 'span': glc_t1 - 2500 + 3, 'fc':(1,1,1,0)},
        {'start': glc_t1, 'span': -10, 'fc':'#6495ED'},
        {'start': glc_t1, 'span': glc_t2 - glc_t1 + 3, 'fc':(1,1,1,0)},
        {'start': glc_t2, 'span': -10, 'fc':'#6495ED'},
        {'start': glc_t2, 'span': glc_t3 - glc_t2 + 3, 'fc':(1,1,1,0)},
        {'start': glc_t3, 'span': -10, 'fc':'#6495ED'},
        {'start': glc_t3, 'span': -75, 'fc':(1,1,1,0)}
    ]

    for box in transvaal_boxes:
        box_ = matplotlib.patches.Rectangle((box['start'],0.4), box['span'], 0.15, 
            facecolor=box['fc'], edgecolor='k')
        axs[4].add_patch(box_)

    for box in huronian_boxes:
        box_ = matplotlib.patches.Rectangle((box['start'],0.1), box['span'], 0.15, 
            facecolor=box['fc'], edgecolor='k')
        axs[4].add_patch(box_)

    y_transv = 0.52 if profiles else 0.41
    y_hurn = 0.495 if profiles else 0.36
    plt.text(0.08, y_transv, 'Transvaal\n   Basin', fontsize=6, 
        transform=plt.gcf().transFigure)
    plt.text(0.085, y_hurn, 'Huronian\n   Basin', fontsize=6, 
        transform=plt.gcf().transFigure)

    # plot proxies. Ages are approximates from Gumsley 2017 Fig 3
    colors = {'MIF-S':'k', 'No MIF-S':'#ad941f', 'Negative Ce anomaly':'#ad941f', 
        'Mn deposits':'#ad941f','Detrital pyrite and uraninite ':'k',
        'Oxidized paleosol':'#ad941f', 'Red beds':'#ad941f','Reduced paleosol':'k'}
    
    markers = {'MIF-S':'*', 'No MIF-S':'*', 'Negative Ce anomaly':'Ce', 
        'Mn deposits':'Mn', 'Detrital pyrite and uraninite ':'o', 
        'Oxidized paleosol':'d', 'Red beds':'Hm', 'Reduced paleosol':'d'}
    
    types = {'MIF-S':'s', 'No MIF-S':'s', 'Negative Ce anomaly':'t', 
        'Mn deposits':'t', 'Detrital pyrite and uraninite ':'s',
        'Oxidized paleosol':'s', 'Red beds':'t', 'Reduced paleosol':'s'}


    labels = {}
    positions = [0.475, 0.175]
    for proxy in proxies.Proxy.unique():
        for i, basin in enumerate(['Transvaal', 'Huronian']):
            data_basin = proxies[proxies.Basin == basin]
            data_basin = data_basin[data_basin.Proxy == proxy]
            if types[proxy] == 's':
                plot = plt.scatter(data_basin['Assigned date Mya'], 
                    [positions[i]] * len(data_basin['Assigned date Mya']),
                    marker=markers[proxy], color=colors[proxy], zorder=10, s=12)
                labels[proxy] = plot
            else:
                for index, row in data_basin.iterrows():
                    text = plt.text(x=row['Assigned date Mya'], y=positions[i]-0.02,
                            s=markers[proxy], color=colors[proxy], fontsize=5)
                    labels[proxy] = text

    # Add question marks
    proxies_qm = proxies[proxies['has question mark']]
    for i, row in proxies_qm.iterrows():
        if row.Basin == 'Huronian':
            plt.text(x=row['Assigned date Mya'] -4, y=0.175-0.02, s='?', fontsize=5)
        elif row.Basin == 'Transvaal':
            plt.text(x=row['Assigned date Mya'] -4, y=0.475-0.02, s='?', fontsize=5)

    #add legend
    class TextHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,xdescent, ydescent,
                            width, height, fontsize,trans):
            h = copy.copy(orig_handle)
            h.set_position((width/2.,height/2.))
            h.set_transform(trans)
            h.set_ha("center");h.set_va("center")
            fp = orig_handle.get_font_properties().copy()
            fp.set_size(fontsize)
            # uncomment the following line, 
            # if legend symbol should have the same size as in the plot
            h.set_font_properties(fp)
            return [h]
    handlermap = {type(text) : TextHandler()}
    axs[4].legend(labels.values(), labels.keys(), handler_map=handlermap,
        fontsize=4, ncol=2, frameon=0)
    
    # formation names
    plt.text(glc_t1 + 2, 0.4 + 0.18, 'Mkg', color='b', fontsize=5)
    plt.text(glc_t1 + 2, 0.1 + 0.18, 'RL', color='b', fontsize=5)
    plt.text(glc_t2 + 2, 0.4 + 0.18, 'Dtld', color='b', fontsize=5)
    plt.text(glc_t2 + 2, 0.1 + 0.18, 'Br', color='b', fontsize=5)
    plt.text(glc_t3 + 2, 0.4 + 0.18, 'Roo', color='b', fontsize=5)
    plt.text(glc_t3 + 2, 0.1 + 0.18, 'Gwd', color='b', fontsize=5)
    plt.text(glc_t4 + 2, 0.4 + 0.18, 'Rie', color='b', fontsize=5)

    # Age constraints
    age_lower_timeball_hill = 2316 #(Hannah et al., 2004)
    age_mak = 2423 # Senger et. al. 2023
    age_upper_timeball_hill = 2256 # Rasmussen et al. (2013).
    axs[4].vlines(x=age_lower_timeball_hill, ymin=0.67, ymax=0.78, linewidth=0.8,
        color='k')
    axs[4].hlines(y=0.78, xmin=age_lower_timeball_hill-7, 
        xmax=age_lower_timeball_hill+7, linewidth=0.8,color='k')
    plt.text(age_lower_timeball_hill+10, 0.8, r'$2316 \pm 7^4$', fontsize=4.5)
    axs[4].vlines(x=age_mak, ymin=0.67, ymax=0.78, linewidth=0.8,
        color='k')
    axs[4].hlines(y=0.78, xmin=age_mak-1, xmax=age_mak+1, linewidth=0.8,
        color='k')
    plt.text(age_mak+20, 0.8, r'$2423 \pm 1^1$', fontsize=4.5)
    axs[4].vlines(x=age_upper_timeball_hill, ymin=0.67, ymax=0.78, linewidth=0.8,
        color='k')
    axs[4].hlines(y=0.78, xmin=age_upper_timeball_hill-12, 
        xmax=age_upper_timeball_hill+12, linewidth=0.8,color='k')
    plt.text(age_upper_timeball_hill+10, 0.8, r'$2256 \pm 12^5$', fontsize=4.5)
    # age_dui_max = 2424 #+-12 Schröder et. al 2016
    # age_dui_min = 2342 #± 18 Zeh et. al. 2020
    axs[4].vlines(x=2370, ymin=0.67, ymax=0.78, linewidth=0.8, color='k')
    axs[4].hlines(y=0.78, xmin=2420, xmax=2342, linewidth=0.8, color='k')
    plt.text(2370+30, 0.8, r'$2424-2342^{2,3}$?', fontsize=4.5)
    
    axs[4].axis('off')

    # Panel D: ch4 mixing ratio
    # ----------------------------------------------------------------------
    plt.sca(axs[5])
    axs[5].invert_xaxis()
    plt.setp(axs[5].get_xticklabels(), visible=False)
    plt.xlim([2520, 1980])
    plt.plot(output_case1.time, output_case1.CH4, color='k', lw=1, label=r'$CH_4$')
    plt.plot(output_case2.time, output_case2.CH4, color='k', lw=1, ls=':', label=r'$CH_4$')
    plt.yscale('log')
    plt.ylim([5e-7, 1e-3])
    axs[5].text(0.01, 0.9, 'd)', transform=axs[5].transAxes, size=6)
    plt.ylabel(r'$CH_4$ mixing ratio', fontsize='6')
    for gl_time in glaciation_times:
        axs[5].axvspan(gl_time, gl_time-10, alpha=0.6, color='#6495ED')

    # Panel E: columns
    # ----------------------------------------------------------------------
    plt.sca(axs[6])
    axs[6].invert_xaxis()
    plt.xlim([2520, 1980])
    plt.plot(output_case1.time, output_case1.S8AER_col, color='#f2991b', lw=1, 
        label=r'$S_8$', zorder=10)
    plt.plot(output_case1.time, output_case1.O3_col, color='k', lw=1, label=r'$O_3$',
        zorder=5)
    plt.plot(output_case2.time, output_case2.S8AER_col, color='#f2991b', lw=1,
        zorder=10, ls=':')
    plt.plot(output_case2.time, output_case2.O3_col, color='k', lw=1,
        zorder=5, ls=':')
    plt.yscale('log')
    axs[6].text(0.01, 0.9, 'e)', transform=axs[6].transAxes, size=6)
    plt.ylabel('Column\n' + r'($/cm^2$)', fontsize='6')
    plt.ylim([1e5, 1e19])
    plt.legend(fontsize=5, loc='lower right', frameon=0)
    plt.xlabel('Time (Mya)',fontsize=6)
    for gl_time in glaciation_times:
        axs[6].axvspan(gl_time, gl_time-10, alpha=0.6, color='#6495ED')

    if profiles:
        axs[7].set_visible(False)
        # read model output
        output = pd.read_csv('reduced_model_output/atmospheric_profiles.csv')

        # change time scale
        my = 365*24*60*60*1e6
        output['time'] = np.abs(output['time']/my - 2500)

        output_case1 = output[output.O2_flux == 1.8e12]
        output_case2 = output[output.O2_flux == 2.2e12]

        # profile_points_age = [2460, 2425, 2415.3]
        profile_points_age = output.time.unique()
        alt = np.arange(0.25, 100, 0.5)
        species = ['O2', 'O3', 'CH4', 'H2O', 'OH']
        labels = [r'$O_2$', r'$O_3$', r'$CH_4$', r'$H_2O$', r'$OH$']
        colors = ['k', '#6495ED', '#de626d']
        titles = ['2460Ma Temperate climate (290K)',
            '2425Ma Glacial climate (250K)',
            '2420Ma Hot-moist grenhouse climate (320K)']
        letters = ['f)', 'g)', 'h)', 'i)', 'j)']

        for i, sp in enumerate(species):
            plt.sca(axs[i+8])
            if i in [2,3]:
                axs[i+8].text(0.89, 0.90, letters[i], transform=axs[i+8].transAxes, size=6)
            else:
                axs[i+8].text(0.04, 0.90, letters[i], transform=axs[i+8].transAxes, size=6)

            for j, age in enumerate(profile_points_age):
                output_age_c1 = output_case1[output_case1.time == age]
                output_age_c2 = output_case2[output_case2.time == age]
                plt.plot(output_age_c1[sp], alt, color=colors[j],
                    label=labels[j], lw=0.7, alpha=0.9)
                plt.plot(output_age_c2[sp], alt, color=colors[j], ls=':',
                    lw=0.7, alpha=0.9)
            plt.xscale('log')
            plt.xlabel(f'{labels[i]} Mixing ratio', fontsize=6)
            if i == 0:
                plt.ylabel('Altitude (km)', fontsize=6)
                h1, = plt.plot([], [], color=colors[0], lw=0.7, alpha=0.9)
                h2, = plt.plot([], [], color=colors[1], lw=0.7, alpha=0.9)
                h3, = plt.plot([], [], color=colors[2], lw=0.7, alpha=0.9)
                h4, = plt.plot([], [], color='gray', lw=0.7, alpha=0.9)
                h5, = plt.plot([], [], color='gray', ls=':', lw=0.7, alpha=0.9)

                handles1 = [h4, h5]
                labels1 = [r'$1.8 \times 10^{12} /cm^2/s$', r'$2.2 \times 10^{12} /cm^2/s$',]
                
                handles2 = [h1, h2, h3]
                labels2 = titles
                
                axbox = axs[10].get_position()
                # create legend
                # leg1 = fig.legend(handles1, labels1, loc='lower center', ncol=2,
                #     fontsize=6,
                #     bbox_to_anchor=[axbox.x0 + 0.5*axbox.width, axbox.y0-0.09], 
                #     bbox_transform=fig.transFigure, frameon=False, title_fontsize=8)
                # c = leg1.get_children()[0]
                # hpack = c.get_children()[1]
                # c._children = [hpack]

                leg2 = fig.legend(handles2, labels2, loc='lower center', ncol=3,
                    fontsize=6,
                    bbox_to_anchor=[axbox.x0 + 0.5*axbox.width, axbox.y0-0.07], 
                    bbox_transform=fig.transFigure, frameon=False, title_fontsize=8)
                c = leg2.get_children()[0]
                hpack = c.get_children()[1]
                c._children = [hpack]
            else:
                plt.yticks(color='#ffffff00')
        
    plt.subplots_adjust(hspace=0, wspace=0.08)

    if o2_flux_increase:
        plt.savefig(f'o2_flux_linear_increase_evolution.pdf', bbox_inches='tight')
    elif ri_flux_decrease:
        plt.savefig(f'ri_flux_linear_decrease_evolution.pdf', bbox_inches='tight')
    else:
        if profiles:
            plt.savefig(f'o2_flux_constant_evolution_profiles.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'o2_flux_constant_evolution.pdf', bbox_inches='tight')


def profiles():
    '''
    Plot atmospheric profiles at three points in time.
    '''
    # read model output
    output = pd.read_csv('reduced_model_output/atmospheric_profiles.csv')

    # change time scale
    my = 365*24*60*60*1e6
    output['time'] = np.abs(output['time']/my - 2500)

    output_case1 = output[output.O2_flux == 1.8e12]
    output_case2 = output[output.O2_flux == 2.2e12]

    # profile_points_age = [2460, 2425, 2415.3]
    profile_points_age = output.time.unique()
    alt = np.arange(0.25, 100, 0.5)
    species = ['O2', 'O3', 'CH4', 'H2O', 'OH']
    labels = [r'$O_2$', r'$O_3$', r'$CH_4$', r'$H_2O$', r'$OH$']
    colors = ['k', 'r', 'orange', 'b', 'cyan']
    titles = ['2460Ma\n Temperate climate (290K)',
        '2425Ma\n Glacial climate (250K)',
        '2420Ma\n Hot-moist grenhouse climate (320K)']
    letters = ['a)', 'b)', 'c)']
    fig, axs = plt.subplots(1, 3, figsize=(a4_x * 0.8, a4_y * 0.3), sharey='col', sharex='row')

    for i, age in enumerate(profile_points_age):
        plt.sca(axs[i])
        axs[i].title.set_text(titles[i])
        axs[i].text(0.04, 0.9, letters[i], transform=axs[i].transAxes, size=6)
        axs[i].title.set_size(7)
        output_age_c1 = output_case1[output_case1.time == age]
        output_age_c2 = output_case2[output_case2.time == age]
        for j, sp in enumerate(species):
            plt.plot(output_age_c1[sp], alt, color=colors[j], label=labels[j], lw=0.7, alpha=0.9)
            plt.plot(output_age_c2[sp], alt, color=colors[j], ls=':', lw=0.7, alpha=0.9)
        plt.xscale('log')
        plt.xlim([1e-15,1e-1])
        plt.xlabel('Mixing ratio', fontsize=6)
        if i == 0:
            plt.ylabel('Altitude (km)', fontsize=6)
    
            h1, = plt.plot([], [], color=colors[0], lw=0.7, alpha=0.9)
            h2, = plt.plot([], [], color=colors[1], lw=0.7, alpha=0.9)
            h3, = plt.plot([], [], color=colors[2], lw=0.7, alpha=0.9)
            h4, = plt.plot([], [], color=colors[3], lw=0.7, alpha=0.9)
            h5, = plt.plot([], [], color=colors[4], lw=0.7, alpha=0.9)
            h6, = plt.plot([], [], color='gray', lw=0.7, alpha=0.9)
            h7, = plt.plot([], [], color='gray', ls=':', lw=0.7, alpha=0.9)

            handles1 = [h6, h7]
            labels1 = [r'$1.8 \times 10^{12} /cm^2/s$', r'$2.2 \times 10^{12} /cm^2/s$',]
            
            handles2 = [h1, h2, h3, h4, h5]
            labels2 = [r'$O_2$', r'$O_3$', r'$CH_4$', r'$H_2O$', r'$OH$']
            
            axbox = axs[1].get_position()
            # create legend
            leg1 = fig.legend(handles1, labels1, loc='lower center', ncol=2,
                fontsize=6,
                bbox_to_anchor=[axbox.x0 + 0.5*axbox.width, axbox.y0-0.18], 
                bbox_transform=fig.transFigure, frameon=False, title_fontsize=8)
            c = leg1.get_children()[0]
            hpack = c.get_children()[1]
            c._children = [hpack]

            leg2 = fig.legend(handles2, labels2, loc='lower center', ncol=5,
                fontsize=6,
                bbox_to_anchor=[axbox.x0 + 0.5*axbox.width, axbox.y0-0.22], 
                bbox_transform=fig.transFigure, frameon=False, title_fontsize=8)
            c = leg2.get_children()[0]
            hpack = c.get_children()[1]
            c._children = [hpack]
    plt.savefig('profiles.pdf', bbox_inches='tight')

def profiles_by_species():
    '''
    Plot atmospheric profiles at three points in time.
    '''
    # read model output
    output = pd.read_csv('reduced_model_output/atmospheric_profiles.csv')

    # change time scale
    my = 365*24*60*60*1e6
    output['time'] = np.abs(output['time']/my - 2500)

    output_case1 = output[output.O2_flux == 1.8e12]
    output_case2 = output[output.O2_flux == 2.2e12]

    # profile_points_age = [2460, 2425, 2415.3]
    profile_points_age = output.time.unique()
    alt = np.arange(0.25, 100, 0.5)
    species = ['O2', 'O3', 'CH4', 'H2O', 'OH']
    labels = [r'$O_2$', r'$O_3$', r'$CH_4$', r'$H_2O$', r'$OH$']
    colors = ['k', '#109fe5', 'r']
    titles = ['2460Ma Temperate climate (290K)',
        '2425Ma Glacial climate (250K)',
        '2420Ma Hot-moist grenhouse climate (320K)']
    letters = ['a)', 'b)', 'c)', 'd)', 'e)']
    fig, axs = plt.subplots(1, 5, figsize=(a4_x * 0.8, a4_y * 0.25), sharey=True)

    for i, sp in enumerate(species):
        plt.sca(axs[i])
        if i in [2,3]:
            axs[i].text(0.89, 0.95, letters[i], transform=axs[i].transAxes, size=6)
        else:
            axs[i].text(0.04, 0.95, letters[i], transform=axs[i].transAxes, size=6)

        for j, age in enumerate(profile_points_age):
            output_age_c1 = output_case1[output_case1.time == age]
            output_age_c2 = output_case2[output_case2.time == age]
            plt.plot(output_age_c1[sp], alt, color=colors[j], label=labels[j], lw=0.7, alpha=0.9)
            plt.plot(output_age_c2[sp], alt, color=colors[j], ls=':', lw=0.7, alpha=0.9)
        plt.xscale('log')
        # plt.xlim([1e-15,1e-1])
        plt.xlabel(f'{labels[i]} Mixing ratio', fontsize=6)
        if i in [0]:
            plt.ylabel('Altitude (km)', fontsize=6)
        if i == 0:
            h1, = plt.plot([], [], color=colors[0], lw=0.7, alpha=0.9)
            h2, = plt.plot([], [], color=colors[1], lw=0.7, alpha=0.9)
            h3, = plt.plot([], [], color=colors[2], lw=0.7, alpha=0.9)
            h4, = plt.plot([], [], color='gray', lw=0.7, alpha=0.9)
            h5, = plt.plot([], [], color='gray', ls=':', lw=0.7, alpha=0.9)

            handles1 = [h4, h5]
            labels1 = [r'$1.8 \times 10^{12} /cm^2/s$', r'$2.2 \times 10^{12} /cm^2/s$',]
            
            handles2 = [h1, h2, h3]
            labels2 = titles
            
            axbox = axs[2].get_position()
            # create legend
            leg1 = fig.legend(handles1, labels1, loc='lower center', ncol=2,
                fontsize=6,
                bbox_to_anchor=[axbox.x0 + 0.5*axbox.width, axbox.y0-0.24], 
                bbox_transform=fig.transFigure, frameon=False, title_fontsize=8)
            c = leg1.get_children()[0]
            hpack = c.get_children()[1]
            c._children = [hpack]

            leg2 = fig.legend(handles2, labels2, loc='lower center', ncol=3,
                fontsize=6,
                bbox_to_anchor=[axbox.x0 + 0.5*axbox.width, axbox.y0-0.19], 
                bbox_transform=fig.transFigure, frameon=False, title_fontsize=8)
            c = leg2.get_children()[0]
            hpack = c.get_children()[1]
            c._children = [hpack]
    plt.subplots_adjust(wspace=0.1)
    plt.savefig('profiles_by_sp_1.pdf', bbox_inches='tight')



parameter_space()
results_with_proxies(o2_flux_increase=False)
results_with_proxies(o2_flux_increase=True)
results_with_proxies(ri_flux_decrease=True)
profiles()
results_with_proxies(o2_flux_increase=False, profiles=True)
profiles_by_species()
"""Functions for plotting brainvoyager voxeltimecourse files 
functions created by Jorie van Haren (2022), and tested on brainvoyaer version 22.2. 
for any help email jjg.vanharen@maastrichtuniversity.nl"""

# import things we need
import numpy as np
import os
import re
import os
import scipy
import scipy.ndimage

from os.path import join
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import animation

import bvbabel

from bv_preproc.utils import (print_f, prefix,
                              target_dir, preproc_filenames,
                              _regex_preproc_filenames)
from bv_preproc.voxeltimecourse import vtc_names


## PLOTTING FUNCTIONS

def plot_vtcs(input_dir,
              slicor = True,  # include slice scan correction naming conv
              motcor = True,  # include motion correction naming conv
              hpfil = True,   # include highpass filter naming conv
              tpsmo = True,   # include temporal smoothing naming conv):
              topup = True,
              bv = None): 
    """plot difference plot, proccessed data agains refference volume (default run 1 volume 1)
    input: directory where `fmr` files are located,
    optional input: plot_sbref=True, plot sbref difs allongside the run plots, 
                    please set to false if no sbref is present
    returns figure and ax, plot"""
    # search trough dircreate_fmr(func_dict, input_folder, target_folder, currun, skip_volumes_end)      
    # function creates fmr file for sellected runectory for preprocessed files
    re_pattern = _regex_preproc_filenames(sbref=2, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, topup=topup)
    dir_items = os.listdir(input_dir)
    dir_match = sorted([s for s in dir_items if re.search(re_pattern, s)])  
    dir_match = vtc_names(dir_match)

    # get rumber of runs in dir
    n_runs = len(dir_match)
    
    # inform on plotting params
    print_f('\nPlotting VTCs for {} runs\n  directory: {}' \
            '\n  slice scan correction: {}\n  motion correction: {}\n  ' \
            'highpass filter: {}\n  temporal smoothing:  {}'.format(n_runs,
                                                                    input_dir,
                                                                    slicor,
                                                                    motcor,
                                                                    hpfil,
                                                                    tpsmo), bv=bv)

    # initalize figure
    fig, ax = plt.subplots(2, 
                           n_runs, 
                           sharex=True,
                           figsize=(n_runs*2, 4), gridspec_kw={'height_ratios': [3, 1]})

    # loop over runs to plot
    for currun in range(len(dir_match)):
        
        # load current fmr file 
        header, curvtc = bvbabel.vtc.read_vtc(join(input_dir, dir_match[currun]))

        # calculate normalized difference plot
        calc_fmr_tr = np.mean(curvtc, axis=3)[:,:,round(curvtc.shape[2]/2)]
        calc_fmr_co = np.mean(curvtc, axis=3)[:,round(curvtc.shape[1]/2),:]
        ax[0, currun].imshow(np.rot90(calc_fmr_tr), cmap='gray')
        ax[1, currun].imshow(np.rot90(calc_fmr_co), cmap='gray')

        # general plotting settings
        ax[0, currun].set_title('Run {}'.format(currun+1), fontsize=18)    # set run as title
        ax[0, currun].axes.xaxis.set_visible(False)
        ax[0, currun].axes.yaxis.set_visible(False)
        ax[1, currun].axes.xaxis.set_visible(False)
        ax[1, currun].axes.yaxis.set_visible(False)

    # adjust overall and shared plot parameters
    ax[0, 0].set_ylabel('Axial', fontsize=14)
    ax[1, 0].set_ylabel('Coronal', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.suptitle('Voxel Time Courses - Mean over time\n' \
                 'Middle volumes, middle slices - every run', fontsize=22)
    plt.tight_layout()
    return(ax, fig)

"""Functions for plotting brainvoyager functional files and preprocessing functional data
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
import bv_preproc.bvbabel_light as bvbabel_light


## PLOTTING FUNCTIONS

def read_motcor_log(filename, bv=None):
    """read a logfile, parse it and get out motion correction parameters"""
    # update user
    print_f('\nParsing motor correction logfiles available within: {}'.format(filename), bv=bv)
    
    # open current document
    with open(filename, 'r') as f:
       # Read the file contents and generate a list with each line
       lines = f.readlines()
       
    # set search patterns for parsing
    corpattern = lambda p : r'(?<={}: )[-0-9.]+'.format(p)
    nr_vols = int(re.search(r'(?<=volume:)[ \t]+[0-9]+', lines[-1])[0])

    # predefine matrix
    coords = np.empty((nr_vols, 6))
    coords[:] = np.nan

    for line in lines:

        # match regex
        match = re.search(r'->', line)
        if match:
            # get back saved coordinate shifts
            dx = float(re.search(corpattern('dx'), line)[0])
            dy = float(re.search(corpattern('dy'), line)[0])
            dz = float(re.search(corpattern('dz'), line)[0])
            rx = float(re.search(corpattern('rx'), line)[0])
            ry = float(re.search(corpattern('ry'), line)[0])
            rz = float(re.search(corpattern('rz'), line)[0])

            # save coordinate dictionary
            coords[int(re.search(r'(?<=volume:)[ \t]+[0-9]+', line)[0])-1,
                   :] = [dx, dy, dz, rx, ry, rz]

    # return coords array
    return(coords)


def plot_motcor_graphs(input_dir, plot_sbref=True, bv=None):
    """plot motion correction graphs for a given directory
    input: directory where `3DMC.log` files are located,
    optional input: plot_sbref=True, plot sbref motion correction allongside the graphs, 
                    please set to false if no sbref is present
    returns figure and ax, plot"""
    # search trough directory for log files
    dir_items = os.listdir(input_dir)
    dir_match = sorted([s for s in dir_items if re.search(r'[0-9]_FMR_(?=.*3DMC)[a-zA-Z0-9_]+.log$', s)]) # regex for run log files
    sbref_match = [[s for s in dir_items if re.search(r'SBRef_FMR_3DMC.log$', s)][0]]  # regex search for sbref log file
    if not sbref_match: sbref_match = [' ']     # to circumvent length error add empy if non existing
    all_match = sbref_match + dir_match         # combine match lists (first sbref, then runs)

    # get rumber of runs in dir
    n_runs = len(dir_match)
    
    # update user
    print_f('\nPlotting motion correction graphs for {} runs\n  directory: {}\n  plot sbref: {}'.format(n_runs, 
                                                                                                        input_dir, 
                                                                                                        plot_sbref), bv=bv)

    # adjust settings for when we want to plot sbref and when not
    if plot_sbref: 
        n_subplots = n_runs+1
        ratios = [1] + ([6] * n_runs)
        start = 0
    else:
        n_subplots = n_runs
        ratios = ([1] * n_runs)
        start = 1

    # initalize figure
    fig, ax = plt.subplots(1, 
                           n_subplots, 
                           sharey=True, 
                           figsize=(n_runs*2.5, 3.5), gridspec_kw={'width_ratios':ratios})

    # loop over runs to plot
    for currun in np.arange(start,len(all_match)):
        # adjust subidx for when we plot sbref or nto
        subidx = currun - start

        # plot sbref adjustment
        if currun == 0:       
            coords = np.repeat(read_motcor_log(join(input_dir, sbref_match[0]), bv=bv), 2, axis=0)  # repeat single value to make line
            ax[subidx].set_title('SBRef', fontsize=18)                              # set title for plotting

        elif currun > 0: 
            coords = read_motcor_log(join(input_dir, all_match[currun]), bv=bv)                     # read log file and get coords
            ax[subidx].set_title('Run {}'.format(currun), fontsize=18)              # set run as title
            ax[subidx].set_xlabel('Volume', fontsize=16)                            # label x axis

        # genral plotting settings
        ax[subidx].plot(coords, lw=2.5)                                             # plot actual coords
        for axx in np.arange(np.floor(np.amin(coords)), np.ceil(np.amax(coords))+1, 1):
            ax[subidx].axhline(y=axx, color='grey', linestyle='--', lw=2)           # set lines for every whole number
        ax[subidx].tick_params(axis='x', which='major', labelsize=16)               # set ticksizes x 
        ax[subidx].tick_params(axis='y', which='major', labelsize=16)               # and y

    # put legend in last subplot
    ax[-1].legend(['dx', 'dy', 'dz', 'rx', 'ry', 'rz'], fontsize=14, fancybox=True, shadow=True, bbox_to_anchor=(1, 1))

    # adjust overall and shared plot parameters
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(np.floor(np.amin(coords))-0.1, np.ceil(np.amax(coords))+0.1)
    plt.suptitle('Motion Parameters', fontsize=22)
    plt.tight_layout()
    
    return(ax, fig)


def plot_preproc_ref(input_dir, plot_sbref=True,
                     slicor = True,  # include slice scan correction naming conv
                     motcor = True,  # include motion correction naming conv
                     hpfil = True,   # include highpass filter naming conv
                     tpsmo = True,   # include temporal smoothing naming conv):
                     refrun = 1,     # sellect refference run
                     refvol = 0,     # sellect refference volume (0 is first)
                     bv=None):    
    """plot difference plot, unprocessed data minus processed data
    input: directory where `fmr` files are located,
    optional input: plot_sbref=True, plot sbref results allong run results, 
                    please set to false if no sbref is present
    returns figure and ax, plot"""

    # search trough directory for preprocessed files
    re_pattern = _regex_preproc_filenames(sbref=2, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo)
    dir_items = os.listdir(input_dir)
    dir_match = sorted([s for s in dir_items if re.search(re_pattern, s)])
    
    # search trough directory for unproccessed files
    ref_match = sorted([s for s in dir_items if re.search('run{}_FMR.fmr'.format(refrun), s)])

    # get rumber of runs in dir
    n_runs = len(dir_match)

    
    # inform on plotting params
    print_f('\nPlotting differences preprocced data vs refference slice data for {} runs\n  directory: {}\n  plot sbref: {}' \
             '\n  slice scan correction: {}\n  motion correction: {}\n  highpass filter: {}\n  temporal smoothing:  {}' \
             '\n\n  refference run: {}\n  refference volume: {}' .format(n_runs, 
                                                                         input_dir, 
                                                                         plot_sbref, 
                                                                         slicor, 
                                                                         motcor, 
                                                                         hpfil, 
                                                                         tpsmo, 
                                                                         refrun, 
                                                                         refvol), bv=bv)
    
    # get sbref data if wanted
    if plot_sbref: 
        n_runs+=1 # get extra window if we plot sbref
        re_pattern = _regex_preproc_filenames(sbref=3, slicor=False, motcor=True, hpfil=False, tpsmo=False)
        sbref_match = [s for s in dir_items if re.search(re_pattern, s)]

    # initalize figure
    fig, ax = plt.subplots(2, 
                           n_runs, 
                           sharex=True,
                           figsize=(n_runs*2, 4), gridspec_kw={'height_ratios': [3, 1]})

    # already get the refference slice
    header     = bvbabel_light.read_fmr_header(join(input_dir, ref_match[0]))
    curfmr_ref = bvbabel_light.read_stc_single_volume('{}stc'.format(join(input_dir, ref_match[0])[:-3]), 
                                                      refvol, 
                                                      header['NrOfSlices'], 
                                                      header['NrOfVolumes'], 
                                                      header['ResolutionX'],
                                                      header['ResolutionY'],
                                                      data_type= header['DataType'])
    
    curfmr_ref_tr = _norm_array(curfmr_ref[:,:,round(header['NrOfSlices']/2)])
    curfmr_ref_co = _norm_array(curfmr_ref[:,round(header['ResolutionX']/2),:])

    # loop over runs to plot
    for currun in range(len(dir_match)):
        
        # load current fmr file 
        curfmr_pro = bvbabel_light.read_stc_single_volume('{}stc'.format(join(input_dir, dir_match[currun])[:-3]), 
                                                      round(header['NrOfVolumes']/2), 
                                                      header['NrOfSlices'], 
                                                      header['NrOfVolumes'], 
                                                      header['ResolutionX'],
                                                      header['ResolutionY'],
                                                      data_type= header['DataType'])

        # calculate normalcurfil(input_dir, prefix(pp,ses))ized difference plot
        calc_fmr_tr = curfmr_ref_tr - _norm_array(curfmr_pro[:,:,round(header['NrOfSlices']/2)])
        calc_fmr_co = curfmr_ref_co - _norm_array(curfmr_pro[:,round(header['ResolutionX']/2),:])
        ax[0, currun].imshow(np.rot90(calc_fmr_tr), cmap='gray')
        ax[1, currun].imshow(np.rot90(calc_fmr_co), cmap='gray')

        # general plotting settings
        ax[0, currun].set_title('Run {}'.format(currun+1), fontsize=18)              # set run as title
        ax[0, currun].axes.xaxis.set_visible(False)
        ax[0, currun].axes.yaxis.set_visible(False)
        ax[1, currun].axes.xaxis.set_visible(False)
        ax[1, currun].axes.yaxis.set_visible(False)

    # do the same for sbref if we want to plot this
    if plot_sbref:
 
        # load current fmr file 
        _, curfmr_pro = bvbabel.fmr.read_fmr(join(input_dir, sbref_match[0]))

        # calculate normalized difference plot
        calc_fmr_tr = curfmr_ref_tr - np.squeeze(_norm_array(curfmr_pro[:,:,round(header['NrOfSlices']/2)]))
        calc_fmr_co = curfmr_ref_co - np.squeeze(_norm_array(curfmr_pro[:,round(header['ResolutionX']/2),:]))
        ax[0, -1].imshow(np.rot90(calc_fmr_tr), cmap='gray')
        ax[1, -1].imshow(np.rot90(calc_fmr_co), cmap='gray')

        # general plotting settings
        ax[0, -1].set_title('SBRef', fontsize=18)              # set run as title
        ax[0, -1].axes.xaxis.set_visible(False)
        ax[0, -1].axes.yaxis.set_visible(False)
        ax[1, -1].axes.xaxis.set_visible(False)
        ax[1, -1].axes.yaxis.set_visible(False)

    # adjust overall and shared plot parameters
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.suptitle('Difference preprocessed vs refference slice functional images\nDisplayed for every run' \
                 ' - less visual structure is better', fontsize=22)
    plt.tight_layout()
    return(ax, fig)
    

def plot_preproc_difs(input_dir, plot_sbref=True,
                     slicor = True,  # include slice scan correction naming conv
                     motcor = True,  # include motion correction naming conv
                     hpfil = True,   # include highpass filter naming conv
                     tpsmo = True,   # include temporal smoothing naming conv):
                     bv = None): 
    """plot difference plot, proccessed data agains refference volume (default run 1 volume 1)
    input: directory where `fmr` files are located,
    optional input: plot_sbref=True, plot sbref difs allongside the run plots, 
                    please set to false if no sbref is present
    returns figure and ax, plot"""
    # search trough dircreate_fmr(func_dict, input_folder, target_folder, currun, skip_volumes_end)      
    # function creates fmr file for sellected runectory for preprocessed files
    re_pattern = _regex_preproc_filenames(sbref=2, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo)
    dir_items = os.listdir(input_dir)
    dir_match = sorted([s for s in dir_items if re.search(re_pattern, s)])  

    # search trough directory for unproccessed files
    re_pattern = _regex_preproc_filenames(sbref=2, slicor=False, motcor=False, hpfil=False, tpsmo=False)
    raw_match = sorted([s for s in dir_items if re.search(re_pattern, s)])

    # get rumber of runs in dir
    n_runs = len(dir_match)
    
    # inform on plotting params
    print_f('\nPlotting differences preprocced data vs raw data for {} runs\n  directory: {}\n  plot sbref: {}' \
            '\n  slice scan correction: {}\n  motion correction: {}\n  ' \
            'highpass filter: {}\n  temporal smoothing:  {}'.format(n_runs,
                                                                    input_dir,
                                                                    plot_sbref,
                                                                    slicor,
                                                                    motcor,
                                                                    hpfil,
                                                                    tpsmo), bv=bv)
                                                                                                                               
    # get sbref data if wanted
    if plot_sbref: 
        n_runs+=1 # get extra window if we plot sbref
        re_pattern = _regex_preproc_filenames(sbref=3, slicor=False, motcor=False, hpfil=False, tpsmo=False)
        sbref_match_raw = [s for s in dir_items if re.search(re_pattern, s)]
        re_pattern = _regex_preproc_filenames(sbref=3, slicor=False, motcor=True, hpfil=False, tpsmo=False)
        sbref_match = [s for s in dir_items if re.search(re_pattern, s)]

    # initalize figure
    fig, ax = plt.subplots(2, 
                           n_runs, 
                           sharex=True,
                           figsize=(n_runs*2, 4), gridspec_kw={'height_ratios': [3, 1]})

    # loop over runs to plot
    for currun in range(len(dir_match)):
        
        # load current fmr file 
        header     = bvbabel_light.read_fmr_header(join(input_dir, dir_match[currun]))
        curfmr_pro = bvbabel_light.read_stc_single_volume('{}stc'.format(join(input_dir, dir_match[currun])[:-3]), 
                                                      round(header['NrOfVolumes']/2), 
                                                      header['NrOfSlices'], 
                                                      header['NrOfVolumes'], 
                                                      header['ResolutionX'],
                                                      header['ResolutionY'],
                                                      data_type= header['DataType'])
        curfmr_raw = bvbabel_light.read_stc_single_volume('{}stc'.format(join(input_dir, raw_match[currun])[:-3]), 
                                                      round(header['NrOfVolumes']/2), 
                                                      header['NrOfSlices'], 
                                                      header['NrOfVolumes'], 
                                                      header['ResolutionX'],
                                                      header['ResolutionY'],
                                                      data_type= header['DataType'])

        # calculate normalized difference plot
        calc_fmr_tr = _norm_array(curfmr_pro[:,:,round(header['NrOfSlices']/2)]) - \
                      _norm_array(curfmr_raw[:,:,round(header['NrOfSlices']/2)])
        calc_fmr_co = _norm_array(curfmr_pro[:,round(header['ResolutionX']/2),:]) - \
                      _norm_array(curfmr_raw[:,round(header['ResolutionX']/2),:])
        ax[0, currun].imshow(np.rot90(calc_fmr_tr), cmap='gray')
        ax[1, currun].imshow(np.rot90(calc_fmr_co), cmap='gray')

        # general plotting settings
        ax[0, currun].set_title('Run {}'.format(currun+1), fontsize=18)    # set run as title
        ax[0, currun].axes.xaxis.set_visible(False)
        ax[0, currun].axes.yaxis.set_visible(False)
        ax[1, currun].axes.xaxis.set_visible(False)
        ax[1, currun].axes.yaxis.set_visible(False)

    # do the same _regex_preproc_filenamesfor sbref if we want to plot this
    if plot_sbref:
 
        # load current fmr file 
        header, curfmr_pro = bvbabel.fmr.read_fmr(join(input_dir, sbref_match[0]))
        _, curfmr_raw = bvbabel.fmr.read_fmr(join(input_dir, sbref_match_raw[0]))

        # calculate normalized difference plot
        calc_fmr_tr = _norm_array(curfmr_pro[:,:,round(header['NrOfSlices']/2)]) - \
                      _norm_array(curfmr_raw[:,:,round(header['NrOfSlices']/2)])
        calc_fmr_co = _norm_array(curfmr_pro[:,round(header['ResolutionX']/2),:]) - \
                      _norm_array(curfmr_raw[:,round(header['ResolutionX']/2),:])
        ax[0, -1].imshow(np.rot90(calc_fmr_tr), cmap='gray')
        ax[1, -1].imshow(np.rot90(calc_fmr_co), cmap='gray')

        # general plotting settings
        ax[0, -1].set_title('SBRef', fontsize=18)  # set run as title
        ax[0, -1].axes.xaxis.set_visible(False)
        ax[0, -1].axes.yaxis.set_visible(False)
        ax[1, -1].axes.xaxis.set_visible(False)
        ax[1, -1].axes.yaxis.set_visible(False)

    # adjust overall and shared plot parameters
    ax[0, 0].set_ylabel('Axial', fontsize=14)
    ax[1, 0].set_ylabel('Coronal', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.suptitle('Difference unprocessed vs preprocessed functional images\n' \
                 'Middle volumes, middle slices - every run', fontsize=22)
    plt.tight_layout()
    return(ax, fig)


def plot_topup(input_dir, plot_sbref=True,
                     slicor = True,  # include slice scan correction naming conv
                     motcor = True,  # include motion correction naming conv
                     hpfil = True,   # include highpass filter naming conv
                     tpsmo = True,   # include temporal smoothing naming conv):
                     bv = None):   
    """plot topup distortion correction results
    input: directory where `.fmr` files are located,
    optional input: plot_sbref=True, plot sbref distorion correction, 
                    please set to false if no sbref is present
    returns figure and ax, plot"""
    # search trough dircreate_fmr(func_dict, input_folder, target_folder, currun, skip_volumes_end)      
    # function creates fmr file for sellected runectory for preprocessed files
    re_pattern = _regex_preproc_filenames(sbref=2, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, topup=False)
    dir_items = os.listdir(input_dir)
    dir_match = sorted([s for s in dir_items if re.search(re_pattern, s)])  

    # search trough directory for unproccessed files
    re_pattern = _regex_preproc_filenames(sbref=2, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, topup=True)
    topup_match = sorted([s for s in dir_items if re.search(re_pattern, s)])

    # get rumber of runs in dir
    n_runs = len(dir_match)
    
    # inform on plotting params
    print_f('\nPlotting differences pre and post topup data for {} runs\n  directory: {}\n  plot sbref: {}' \
            '\n  slice scan correction: {}\n  motion correction: {}\n  ' \
            'highpass filter: {}\n  temporal smoothing:  {}'.format(n_runs, 
                                                                    input_dir, 
                                                                    plot_sbref,
                                                                    slicor, 
                                                                    motcor, 
                                                                    hpfil, 
                                                                    tpsmo), bv=bv)
                                                                                                                               
    # get sbref data if wanted
    if plot_sbref: 
        n_runs+=1 # get extra window if we plot sbref
        re_pattern = _regex_preproc_filenames(sbref=3, slicor=False, motcor=True, hpfil=False, tpsmo=False, topup=False)
        sbref_match = [s for s in dir_items if re.search(re_pattern, s)]
        re_pattern = _regex_preproc_filenames(sbref=3, slicor=False, motcor=True, hpfil=False, tpsmo=False, topup=True)
        sbref_match_topup = [s for s in dir_items if re.search(re_pattern, s)]

    # initalize figure
    fig, ax = plt.subplots(1, 
                           n_runs, 
                           sharex=True,
                           figsize=(n_runs*2, 4))

    # loop over runs to plot
    for currun in range(len(dir_match)):
        
        # load current fmr file         
        header     = bvbabel_light.read_fmr_header(join(input_dir, dir_match[currun]))
        curfmr_pre = bvbabel_light.read_stc_single_volume('{}stc'.format(join(input_dir, dir_match[currun])[:-3]), 
                                                      round(header['NrOfVolumes']/2), 
                                                      header['NrOfSlices'], 
                                                      header['NrOfVolumes'], 
                                                      header['ResolutionX'],
                                                      header['ResolutionY'],
                                                      data_type= header['DataType'])
        curfmr_pro = bvbabel_light.read_stc_single_volume('{}stc'.format(join(input_dir, topup_match[currun])[:-3]), 
                                                      round(header['NrOfVolumes']/2), 
                                                      header['NrOfSlices'], 
                                                      header['NrOfVolumes'], 
                                                      header['ResolutionX'],
                                                      header['ResolutionY'],
                                                      data_type= header['DataType'])

        # get contrast image from topup image
        sob = _norm_array(scipy.ndimage.gaussian_gradient_magnitude(curfmr_pro[:,:,round(header['NrOfSlices']/2)], 
                                                                    sigma=1.2))   # calc sobel for x

        # get only values higher then topup
        mediansob = np.median(sob)
        stdsob = np.std(sob)
        sob[sob <= mediansob + (.5 * stdsob)] = np.nan


        # get normalized raw plot and overlay sob image
        calc_fmr_pre = _norm_array(curfmr_pre[:,:,round(header['NrOfSlices']/2)])
        ax[currun].imshow(np.rot90(calc_fmr_pre), cmap='gray', interpolation = 'sinc', vmin=0, vmax=0.6, alpha=0.8)
        ax[currun].imshow(np.rot90(sob), alpha=0.75, cmap='hot')


        # general plotting settings
        ax[currun].set_title('Run {}'.format(currun+1), fontsize=18)   # set run as title
        ax[currun].axes.xaxis.set_visible(False)
        ax[currun].axes.yaxis.set_visible(False)


    # do the same _regex_preproc_filenamesfor sbref if we want to plot this
    if plot_sbref:
 
        # load current fmr file 
        header, curfmr_pro = bvbabel.fmr.read_fmr(join(input_dir, sbref_match_topup[0]))
        _, curfmr_pre = bvbabel.fmr.read_fmr(join(input_dir, sbref_match[0]))

        # get contrast image from topup image
        sob = _norm_array(scipy.ndimage.gaussian_gradient_magnitude(curfmr_pro[:,:,round(header['NrOfSlices']/2)], 
                                                                    sigma=1.2))    # calc sobel for x

        # get only values higher then topup
        mediansob = np.median(sob)
        stdsob = np.std(sob)
        sob[sob <= mediansob + (.6 * stdsob)] = np.nan

        # calculate normalized difference plot
        calc_fmr_pre = _norm_array(curfmr_pre[:,:,round(header['NrOfSlices']/2)])
        ax[-1].imshow(np.rot90(calc_fmr_pre), cmap='gray', interpolation = 'sinc', vmin=0, vmax=0.6, alpha=0.8)
        ax[-1].imshow(np.rot90(sob), alpha=0.75, cmap='hot')

        # general plotting settings
        ax[-1].set_title('SBRef', fontsize=18)   # set run as title
        ax[-1].axes.xaxis.set_visible(False)
        ax[-1].axes.yaxis.set_visible(False)


    # adjust overall and shared plot parameters
    ax[0].set_ylabel('Axial', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.suptitle('Difference pre- (image) vs post- fsl topup (outline) ' \
                 'functional images\nMiddle volumes, middle slices - every run', fontsize=22)
    plt.tight_layout()
    return(ax, fig)
    

def plot_topup_ani(input_dir, plot_sbref=True,
                     slicor = True,  # include slice scan correction naming conv
                     motcor = True,  # include motion correction naming conv
                     hpfil = True,   # include highpass filter naming conv
                     tpsmo = True,   # include temporal smoothing naming conv):
                     bv = None):   
    """plot topup distortion correction results - animated gif
    input: directory where `.fmr` files are located,
    optional input: plot_sbref=True, plot sbref distorion correction, 
                    please set to false if no sbref is present
    returns figure and ax, plot"""
    # search trough dircreate_fmr(func_dict, input_folder, target_folder, currun, skip_volumes_end)      
    # function creates fmr file for sellected runectory for preprocessed files
    re_pattern = _regex_preproc_filenames(sbref=2, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, topup=False)
    dir_items = os.listdir(input_dir)
    dir_match = sorted([s for s in dir_items if re.search(re_pattern, s)])  

    # search trough directory for unproccessed files
    re_pattern = _regex_preproc_filenames(sbref=2, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, topup=True)
    topup_match = sorted([s for s in dir_items if re.search(re_pattern, s)])

    # get rumber of runs in dir
    n_runs = len(dir_match)
    
    # inform on plotting params
    print_f('\nPlotting animated differences pre and post topup data for {} runs\n  directory: {}\n  plot sbref: {}' \
            '\n  slice scan correction: {}\n  motion correction: {}\n  ' \
            'highpass filter: {}\n  temporal smoothing:  {}'.format(n_runs, 
                                                                    input_dir, 
                                                                    plot_sbref,
                                                                    slicor, 
                                                                    motcor, 
                                                                    hpfil, 
                                                                    tpsmo), bv=bv)
                                                                                                                               
    # get sbref data if wanted
    if plot_sbref: 
        n_runs+=1 # get extra window if we plot sbref
        re_pattern = _regex_preproc_filenames(sbref=3, slicor=False, motcor=True, hpfil=False, tpsmo=False, topup=False)
        sbref_match = [s for s in dir_items if re.search(re_pattern, s)]
        re_pattern = _regex_preproc_filenames(sbref=3, slicor=False, motcor=True, hpfil=False, tpsmo=False, topup=True)
        sbref_match_topup = [s for s in dir_items if re.search(re_pattern, s)]

    # initalize figure
    fig, ax = plt.subplots(1, 
                           n_runs, 
                           figsize=(n_runs*2, 4))

    # loop over runs to plot
    img_list = [dir_match, topup_match]
    sbref_img_list = [sbref_match, sbref_match_topup]
    frames = [] # store generated frames
    for i in range(2):

        ims = []
        for currun in range(len(dir_match)):
        
            # load current fmr file 
            header     = bvbabel_light.read_fmr_header(join(input_dir, img_list[i][currun]))
            curfmr     = bvbabel_light.read_stc_single_volume('{}stc'.format(join(input_dir, img_list[i][currun])[:-3]), 
                                                      round(header['NrOfVolumes']/2), 
                                                      header['NrOfSlices'], 
                                                      header['NrOfVolumes'], 
                                                      header['ResolutionX'],
                                                      header['ResolutionY'],
                                                      data_type= header['DataType'])
        
            calc_fmr = curfmr[:,:,round(header['NrOfSlices']/2)]
            im = ax[currun].imshow(np.rot90(calc_fmr), cmap='gray', interpolation = 'sinc', vmin=0, vmax=1200, alpha=0.8, animated=True)
            ims.append(im)

            # general plotting settings
            ax[currun].set_title('Run {}'.format(currun+1), fontsize=18)              # set run as title
            ax[currun].axes.xaxis.set_visible(False)
            ax[currun].axes.yaxis.set_visible(False)

        # do the same _regex_preproc_filenamesfor sbref if we want to plot this
        if plot_sbref:
 
            # load current fmr file 
            header, curfmr = bvbabel.fmr.read_fmr(join(input_dir, sbref_img_list[i][0]))

            calc_fmr = curfmr[:,:,round(header['NrOfSlices']/2)]
            im = ax[-1].imshow(np.rot90(calc_fmr), cmap='gray', interpolation = 'sinc', vmin=0, vmax=1200, alpha=0.8, animated=True)
            ims.append(im)

            # general plotting settings
            ax[-1].set_title('SBRef', fontsize=18)              # set run as title
            ax[-1].axes.xaxis.set_visible(False)
            ax[-1].axes.yaxis.set_visible(False)
            
        frames.append(ims)

    # adjust overall and shared plot parameters
    ax[0].set_ylabel('Axial', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.suptitle('Difference pre- vs post- fsl topup functional images - animated' \
                 '\nMiddle volumes, middle slices - every run', fontsize=22)
    plt.tight_layout()
    return(ax, fig, frames)
    

# HELPER FUNCTIONS

def _norm_array(a_array, maxval=1):
    """helper function to normalize arrawy"""
    return((a_array/np.max(a_array))*maxval)

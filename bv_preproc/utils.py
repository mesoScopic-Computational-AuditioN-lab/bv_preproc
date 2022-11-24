"""Utility functions for bv_preproc module.
functions created by Jorie van Haren (2022), and tested on brainvoyaer version 22.2. 
for any help email jjg.vanharen@maastrichtuniversity.nl"""

# import things we need
import numpy as np
import os
import re
import pickle
import itertools
from pydicom import dcmread
from os.path import join

## FUNCTIONS

## QUICK ACCESS FUNCTIONS
def save_header_dict(func_dict, location):
    """Save header information (functional or anatomical)
    input dictonary and location+name (full path)"""
    return(pickle.dump( func_dict, open( location, "wb" ) ))

def load_header_dict(location):
    """Load previously saved header information dict (funct, or anat)
    input full path to pickle file"""
    return(pickle.load( open( location, "rb" ) ))

## CLEANUP FUNCTIONS
def cleanup_intermediates(input_dir, pps, sess, first_preproc='all', last_preproc='topup', rem_topup_steps=True, leave_log=True, bv=None):
    """function to search in a directories (loop over participants and sessions) for intermediate files and delete them
    input directory where participant and session folders are located, first_preproc step 
    (default, all leaving only the fully processed step,
    but can be any intermediate step in _get_pp_keys()), last_preproc step (default topup, the step to hold on to),
    remove_topup_steps (remove processing steps of topup, true/false), and leave_log (default true, leaving .log files
    intact)"""
    for pp, ses in itertools.product(pps, sess):
        _ = cleanup_intermediate(join(input_dir, prefix(pp,ses)), first_preproc=first_preproc, last_preproc=last_preproc,
                                 rem_topup_steps=rem_topup_steps, leave_log=leave_log, bv=bv)
    return

def cleanup_intermediate(input_dir, first_preproc='all', last_preproc='topup', rem_topup_steps=True, leave_log=True, bv=None):
    """function to search in a directory for intermediate files and delete them
    input directory where files are located, first_preproc step (default, all leaving only the fully processed step,
    but can be any intermediate step in _get_pp_keys()), last_preproc step (default topup, the step to hold on to),
    remove_topup_steps (remove processing steps of topup, true/false), and leave_log (default true, leaving .log files
    intact)"""
    
    # get keys for preocessing steps naming convention/ add all
    ppkeys = _get_pp_keys()
    ppkeys['all'] = 'run'
    
    # add log to re_pattern if wanted
    if leave_log: re_log = '(?!.*log)'
    else: re_log = ''
    
    # create re pattern
    re_pattern = r'[a-zA-Z0-9_.]+(?=.*{})(?!.*{}){}[a-zA-Z0-9_.]+$'.format(ppkeys[first_preproc], ppkeys[last_preproc], re_log)
    
    # list of items 
    dir_items = os.listdir(input_dir)
    matches = [s for s in dir_items if re.search(re_pattern, s)]
    
    # do same for topup files
    if rem_topup_steps: 
        re_pattern_topup = r'[a-zA-Z0-9_.]+(?=.*(TOPUP_AP|TOPUP_PA|TOPUP_APPA)){}[a-zA-Z0-9_.]+$'.format(re_log)   
        matches = matches + [s for s in dir_items if re.search(re_pattern_topup, s)]

    # update user
    print_f('\nRemoving intermediate files in {}:'.format(input_dir), bv=bv)
    
    # main deletion loop
    for i in matches:
        print_f(' - {} [Deleted]'.format(i), bv=bv)
        os.remove(join(input_dir, i))
    return(matches)

def cleanup_nifis(input_dir, pps, sess, bv=None):
    """function to search in directories (loop over participants and sessions) for nifti files and delete"""
    for pp, ses in itertools.product(pps, sess):
        _ = cleanup_nifi(join(input_dir, prefix(pp,ses)), bv=bv)
    return

def cleanup_nifi(input_dir, bv=None):
    """function to search in a directory for nifti files and delete"""
    # create re pattern
    re_pattern = r'[a-zA-Z0-9_.]+(nii|nii.gz)$'
    # list of items 
    dir_items = os.listdir(input_dir)
    matches = [s for s in dir_items if re.search(re_pattern, s)]
    # update user
    print_f('\nRemoving nifti files in {}:'.format(input_dir), bv=bv)
    # main deletion loop
    for i in matches:
        print_f(' - {} [Deleted]'.format(i), bv=bv)
        os.remove(join(input_dir, i))
    return(matches)

## PATH AND DIRECTORY FUNCTIONS
def prefix(pp, ses):
    """set file prefix naming"""
    return('S{:02d}_SES{}'.format(pp, ses))
    
def target_dir(output_path, pp_dir, bv=None):
    """search for and if non-existing, create pp dir at output path"""
    target_folder = join(output_path, pp_dir)
    # check output path (parent directory)
    if not os.path.isdir(output_path):
        print_f('\nPath: {} does not exist, creating now'.format(output_path), bv)
        os.mkdir(output_path)
    # check target folder
    if not os.path.isdir(target_folder): 
        print_f('\nPath: {} does not exist, creating now'.format(target_folder), bv)
        os.mkdir(target_folder)
    else: 
        print_f('\nPath: {} was found!'.format(target_folder), bv)
    return(target_folder)

def preproc_filenames(input_dir, sbref = 2, # 1: includes sbref (if pres, with params), 2: excludes sbref, 3: only sbrefs
                      slicor = True,  # include slice scan correction naming conv
                      motcor = True,  # include motion correction naming conv
                      hpfil = True,   # include highpass filter naming conv
                      tpsmo = True,  # include temporal smoothing naming conv
                      topup = False, # include topup (native to personal func) naming conv
                      dtype = 'fmr'):
    """function to, given a directory, return preprocessed file names in a list (without full path)"""
    re_pattern = _regex_preproc_filenames(sbref=sbref, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, topup=topup, dtype=dtype)
    dir_items = os.listdir(input_dir)
    return(sorted([s for s in dir_items if re.search(re_pattern, s)]))        

# PRINTING TO LOG FUNCTION
def print_f(input, bv=None):
    """helper function test if we are within a interface that can access bv functions
    then returns either a plain print, or printtolog (within brainvoyager)"""
    if bv: p = bv.print_to_log
    else: p = print
    return(p(input))


# HELPER FUNCTIONS
def _regex_preproc_filenames(sbref = 1, # 1: includes sbref (if pres, with params), 2: excludes sbref, 3: only sbrefs
                            slicor = True,  # include slice scan correction naming conv
                            motcor = True,  # include motion correction naming conv
                            hpfil = True,   # include highpass filter naming conv
                            tpsmo = True,  # include temporal smoothing naming conv
                            topup = False, # include topup (native to personal func) naming conv
                            dtype = 'fmr'):
    """helper function not to be called from outside,
    create regex pattern to find function,
    run number, and preproccessing booleans (on or off)"""
    
    # get back needed string identifiers
    keys = ['slicescan','motion','temporal_highpass','temporal_smooth', 'topup']            # sellect needed keys
    sellect = [slicor, motcor, hpfil, tpsmo, topup]                                       # only sellect those asked for
    isnot_re = ['=' if s else '!' for s in sellect]                                # indicer for true or not in regex lookbehind
    str_ident = [_get_pp_keys().get(key) for key in keys]  # find corresponding str identifiers

    # construct regex / add lookbehind
    pp_pat = lambda i, ppstep : '(?{}.*{})'.format(i, ppstep)
    lookbehind = ''.join([pp_pat(i,p) for (i, p) in zip(isnot_re, str_ident)])
    
    # get re pattern and modify to sellect sbrefs or not
    if sbref == 1: re_pattern = r'FMR{}[a-zA-Z0-9_.]+{}$'.format(lookbehind, dtype)
    elif sbref == 2: re_pattern = r'_(?!.*SBRef)[a-zA-Z0-9_]+FMR{}[a-zA-Z0-9_.]+{}$'.format(lookbehind, dtype)
    elif sbref == 3: re_pattern = r'_(?=.*SBRef)[a-zA-Z0-9_]+FMR{}[a-zA-Z0-9_.]+{}$'.format(lookbehind, dtype)
    return(re_pattern)

def _sort_human(l):
    """sort files in a more human like manner"""
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

def _get_pp_keys():
    """helper function not to be called from outside,
    load keys for preproccessing steps"""
    return({'motion': '3DM',
               'slicescan': 'SC',
               'temporal_highpass': 'THP',
               'temporal_lineartrend': 'LTR',
               'spacial_smooting': 'SD',
               'temporal_smooth': 'TDT',
               'mean_intensity': 'MIA',
               'topup': 'TOPUP'})

"""Functions for creating brainvoyager voxeltimecourse (vtc) files
functions created by Jorie van Haren (2022), and tested on brainvoyaer version 22.2. 
for any help email jjg.vanharen@maastrichtuniversity.nl"""

# import things we need
import numpy as np
import os
import itertools
import re
import pickle
from os.path import join

from bv_preproc.utils import (print_f, prefix,
                              preproc_filenames)

import bvbabel

## HIGHER LEVEL FUNCTIONS

def create_vtc_bulk_all(bv, input_dir, pps, sess, vmr_fn,
                    vtc_list=True,              # default true (automatic), or list of filenames
                    ia_trf_fn=True,             # default true (automatic), or string of initial adjustment file
                    fa_trf_fn=True,             # default true (automatic), or string of fine adjustement file
                    first_vol_bounding_box = True, # use first volume for bounding box dimensions
                    bounding_box_lamb = None,   # lambda for getting bounding box : pp, ses (needed if first_vol_bounding_box:false)
                    vtcspace = 1,               # create vtc in 1: native or 2: acpc space
                    acpc_trf_fn = None,         # if vtcspace is 2, give in acpc fn
                    extended_tal = False,       # use extened tal space for vtc creation (optional)
                    res_to_anat = 2,            # specify spatial resolution
                    interpolation_method = 2,   # interpolation method (0: nearest neighbor, 1: trilinear, 2: sinc)
                    bounding_box_int = 100,     # seperate background voxels from brain voxels
                    data_type = 2,              # 1: interger values, 2: float values
                    trf_lamb = None,            # lambda for specifying trf directory : pp, ses (to create vtc from e.g. other session)
                    fmr_lamb = None,            # lambda for specifying seperate functional directory: pp, ses (to create vtc from e.g. other session)
                    fmr_slicor=True, fmr_motcor=True, fmr_hpfil=True, 
                    fmr_tpsmo=False, fmr_topup=True):   # parameters for automatically getting fmr filelist to create vtcs for
    """create vtc files in native or acpc space given settings loop over all runs
    option to use first run for bounding box or to imput own bounding box
    loop over all participants and sessions,
    warning: be carefull using this extened looping function if you want to use costum bounding boxes"""
    # loop over pp and ses for complete proc pipeline
    for pp, ses in itertools.product(pps, sess):
        # see if we specify a trf and fmr labmda to sellect a specific directory
        if trf_lamb: trf_dir = trf_lamb(pp,ses)                                 # check if lambda function and fill in pp, ses
        else: trf_dir = False
        if fmr_lamb: fmr_dir = fmr_lamb(pp,ses)                                 # check if lambda function and fill in pp, ses
        else: fmr_dir = join(input_dir, prefix(pp, ses))
        if bounding_box_lamb: bounding_box_array = bounding_box_lamb(pp,ses)    # check if lambda function and fill in pp, ses
        else: bounding_box_array = None

        # automatically obtain frm list
        fmr_list = preproc_filenames(fmr_dir, sbref=2, slicor = fmr_slicor,
                                     motcor = fmr_motcor, hpfil = fmr_hpfil, tpsmo = fmr_tpsmo, topup = fmr_topup)

        # bulk create vtcs for this pp and this session
        create_vtc_bulk(bv, join(input_dir, prefix(pp, ses)), vmr_fn, fmr_list, 
                        vtc_list=vtc_list, ia_trf_fn=ia_trf_fn, fa_trf_fn=ia_trf_fn,
                        first_vol_bounding_box = first_vol_bounding_box,
                        bounding_box_array = bounding_box_array,
                        vtcspace = vtcspace,
                        acpc_trf_fn = acpc_trf_fn,
                        extended_tal = extended_tal,
                        res_to_anat = res_to_anat,
                        interpolation_method = interpolation_method,
                        bounding_box_int = bounding_box_int,
                        data_type = data_type,
                        trf_dir = trf_dir,
                        fmr_dir = fmr_dir)
    return

## MAIN FUNCTIONS

def create_vtc(bv, input_dir, vmr_fn, fmr_fn, vtc_fn, ia_trf_fn, fa_trf_fn,
                use_bounding_box,           # whether or not to use bounding box
                bounding_box_array = None,  # array of [[xfrom, xto], [yfrom, yto], [zfrom, zto]]
                vtcspace = 1,               # create vtc in 1: native or 2: acpc space
                acpc_trf_fn = None,         # if vtcspace is 2, give in acpc fn
                extended_tal = False,       # use extened tal space for vtc creation (optional)
                res_to_anat = 2,            # specify spatial resolution
                interpolation_method = 2,   # interpolation method (0: nearest neighbor, 1: trilinear, 2: sinc)
                bounding_box_int = 100,     # seperate background voxels from brain voxels
                data_type = 2,              # 1: interger values, 2: float values
                trf_dir = False,            # seperate trf direcotry (to create vtc from e.g. other session)
                fmr_dir = False):           # seperate functional directory (to create vtc from e.g. other session)
    """create vtc files in native or acpc space given settings"""
    
    # set costum directory for some files if wanted
    if not trf_dir: trf_dir = input_dir
    if not fmr_dir: fmr_dir = input_dir 
    
    # open vmr
    doc_vmr = bv.open_document(join(input_dir, vmr_fn))    
    
    # check parameters
    doc_vmr.vtc_creation_extended_tal_space = extended_tal
    # if bounding box is true sellect it
    if use_bounding_box:
        # set values
        doc_vmr.vtc_creation_use_bounding_box = True
        doc_vmr.vtc_creation_bounding_box_from_x = bounding_box_array[0, 0]
        doc_vmr.vtc_creation_bounding_box_to_x   = bounding_box_array[0, 1]
        doc_vmr.vtc_creation_bounding_box_from_y = bounding_box_array[1, 0]
        doc_vmr.vtc_creation_bounding_box_to_y   = bounding_box_array[1, 1]
        doc_vmr.vtc_creation_bounding_box_from_z = bounding_box_array[2, 0]
        doc_vmr.vtc_creation_bounding_box_to_z   = bounding_box_array[2, 1]
    else:
        doc_vmr.vtc_creation_use_bounding_box = False   
        
    # take full path of files
    fmr_file = join(fmr_dir, fmr_fn)
    vtc_file = join(fmr_dir, vtc_fn)
    ia_trf_fn = join(trf_dir, ia_trf_fn)
    fa_trf_fn = join(trf_dir, fa_trf_fn)

    # do the vtc creation in desired space
    if vtcspace == 1:
        doc_vmr.create_vtc_in_native_space(fmr_file, ia_trf_fn, fa_trf_fn, vtc_file, res_to_anat, interpolation_method,
                                           bounding_box_int, data_type)
    elif vtcspace == 2:
        acpc_trf_fn = join(trf_dir, acpc_trf_fn)
        doc_vmr.create_vtc_in_acpc_space(fmr_file, ia_trf_fn, fa_trf_fn, acpc_trf_fn, vtc_file, res_to_anat, interpolation_method,
                                         bounding_box_int, data_type)
    return

def create_vtc_bulk(bv, input_dir, vmr_fn, fmr_list,
                    vtc_list=True,              # default true (automatic), or list of filenames
                    ia_trf_fn=True,             # default true (automatic), or string of initial adjustment file
                    fa_trf_fn=True,             # default true (automatic), or string of fine adjustement file
                    first_vol_bounding_box = True, # use first volume for bounding box dimensions
                    bounding_box_array = None,  # needed if first_vol_bounding_box is set to false, input own dimensions
                    vtcspace = 1,               # create vtc in 1: native or 2: acpc space
                    acpc_trf_fn = None,         # if vtcspace is 2, give in acpc fn
                    extended_tal = False,       # use extened tal space for vtc creation (optional)
                    res_to_anat = 2,            # specify spatial resolution
                    interpolation_method = 2,   # interpolation method (0: nearest neighbor, 1: trilinear, 2: sinc)
                    bounding_box_int = 100,     # seperate background voxels from brain voxels
                    data_type = 2,              # 1: interger values, 2: float values
                    trf_dir = False,            # seperate trf direcotry (to create vtc from e.g. other session)
                    fmr_dir = False):           # seperate functional directory (to create vtc from e.g. other session)
    """create vtc files in native or acpc space given settings loop over all runs
    option to use first run for bounding box or to imput own bounding box"""

    # set costum directory for some files if wanted
    if not trf_dir: trf_dir = input_dir
    if not fmr_dir: fmr_dir = input_dir 
    
    # get filenames for ia and fa
    if vtc_list: vtc_list = vtc_names(fmr_list)
    if ia_trf_fn: ia_trf_fn = ia_fn(trf_dir)[0]
    if fa_trf_fn: fa_trf_fn = fa_fn(trf_dir)[0]
    if acpc_trf_fn: acpc_trf_fn = acpc_fn(trf_dir)[0]
    
    # loop over the loops
    for i in range(len(fmr_list)):
    
        # print info to log
        print_f('\nCreating vtc for vmr: {}, fmr: {} (ia: {}, fa: {})\n -input dir: {}\n -trf dir: {} \n -fmr dir: {}'.format(vmr_fn, fmr_list[i], ia_trf_fn, fa_trf_fn,
                                                                                                                              input_dir, trf_dir, fmr_dir), bv=bv)

        # use correct bounding box
        if (i == 0) and first_vol_bounding_box:           

            # initial vtc
            create_vtc(bv, input_dir, vmr_fn, fmr_list[i], vtc_list[i], ia_trf_fn, fa_trf_fn, 
                       False, vtcspace=vtcspace, 
                       acpc_trf_fn=acpc_trf_fn, extended_tal=extended_tal,
                       res_to_anat=res_to_anat,interpolation_method=interpolation_method, 
                       bounding_box_int=bounding_box_int, data_type=data_type, trf_dir=trf_dir, fmr_dir=fmr_dir)

            # get bounding box values from vtc header
            vtc_head, _ = bvbabel.vtc.read_vtc(join(fmr_dir, vtc_list[i]))
            bounding_box_array = np.array([[vtc_head['ZStart'], vtc_head['ZEnd']], 
                                           [vtc_head['XStart'], vtc_head['XEnd']], 
                                           [vtc_head['YStart'], vtc_head['YEnd']]])
            print_f('\nBounding Box loaded: x1={}, x2={}, y1={}, y2={}, z1={}, z2={}'.format(vtc_head['XStart'], 
                                                                                            vtc_head['XEnd'], 
                                                                                            vtc_head['YStart'], 
                                                                                            vtc_head['YEnd'], 
                                                                                            vtc_head['ZStart'], 
                                                                                            vtc_head['ZEnd']), bv=bv)
        else: 
            # vtcs for comming runs
            create_vtc(bv, input_dir, vmr_fn, fmr_list[i], vtc_list[i], ia_trf_fn, fa_trf_fn, 
                       True, bounding_box_array=bounding_box_array, vtcspace=vtcspace,
                       acpc_trf_fn=acpc_trf_fn, extended_tal=extended_tal,
                       res_to_anat=res_to_anat,interpolation_method=interpolation_method, 
                       bounding_box_int=bounding_box_int, data_type=data_type, trf_dir=trf_dir, fmr_dir=fmr_dir)
    return(bounding_box_array)

def bounding_box(vtc_path, bv=None):
    """use header information from vtc file and return bounding box array"""
    # get bounding box values from vtc header
    vtc_head, _ = bvbabel.vtc.read_vtc(vtc_path)
    bounding_box_array = np.array([[vtc_head['ZStart'], vtc_head['ZEnd']], 
                                   [vtc_head['XStart'], vtc_head['XEnd']], 
                                   [vtc_head['YStart'], vtc_head['YEnd']]])
    # update user
    print_f('\nBounding Box loaded: x1={}, x2={}, y1={}, y2={}, z1={}, z2={}'.format(vtc_head['XStart'], 
                                                                                    vtc_head['XEnd'], 
                                                                                    vtc_head['YStart'], 
                                                                                    vtc_head['YEnd'], 
                                                                                    vtc_head['ZStart'], 
                                                                                    vtc_head['ZEnd']), bv=bv)
    return(bounding_box_array)


def trf_fn(input_dir, exclude_re, trf_re):
    """find and return trf filename"""
    # set re search patterns 
    re_pattern = trf_re  #get list of all trf files

    # search files
    dir_items = os.listdir(input_dir)
    dir_match = sorted([s for s in dir_items if re.search(re_pattern, s)])

    # then exclude based on exclusion criteria
    dir_match_nomask = [s for s in dir_match if exclude_re not in s]

    return(dir_match_nomask)
  

def ia_fn(input_dir, exclude_re='masked', ia_re='IA.trf$'): 
    return(trf_fn(input_dir, exclude_re, ia_re))
    
def fa_fn(input_dir, exclude_re='masked', fa_re='FA.trf$'): 
    return(trf_fn(input_dir, exclude_re, fa_re))
    
def acpc_fn(input_dir, exclude_re='masked', acpc_re='ACPC.trf$'): 
    return(trf_fn(input_dir, exclude_re, acpc_re))


def vtc_names(fmr_list, old_ft='.fmr$', new_ft='.vtc'):
    return(sorted([re.sub(old_ft, new_ft, s) for s in fmr_list]))

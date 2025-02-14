"""Functions for coregistration brainvoyager functional and anatomical files
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
                              target_dir, preproc_filenames)

# HIGH LEVEL FUNCTIONS - LOOPING OVER PARTICIPANTS

def coregister_bbr_all(bv, input_dir, pps, sess, vmr_fn, fmr_fn_suffix):
    """do bbr coregistration, input directiory, vmr filename, and fmr filename (without prefix)
    the intitral coregistration will create the mash, subsequent will automatically use this mask
    loop over all participants and sessions"""
    for pp, ses in itertools.product(pps, sess):
        fmr_fn = _fmr_fn(join(input_dir, prefix(pp,ses)), fmr_fn_suffix)
        coregister_bbr(bv, join(input_dir, prefix(pp, ses)), vmr_fn, fmr_fn)
    return
    
def copy_mesh_all(bv, input_dir, pps, sess, vmr_fn, masked_suffix='masked_'):
    """copy meshes for other filename, usefull if brainvoyager can use existing meshes (for example in bbr)
    vmr_fn just has to have the same dimensions, and is just uses so bv can actually load the meshes
    loop over all participants and sessions"""
    for pp, ses in itertools.product(pps, sess):
        copy_mesh(bv, join(input_dir, prefix(pp, ses)), vmr_fn, masked_suffix=masked_suffix)
    return

## MAIN FUNCTIONS

def coregister_bbr(bv, input_dir, vmr_fn, fmr_fn, fmr_dir=False):
    """do bbr coregistration, input directiory, vmr filename, and fmr filename
    the intitral coregistration will create the mash, subsequent will automatically use this mask
    optionally put in fmr direcory if coregistration is disired to happen between folders (new files saved in fmr_dir)"""
    
    # if fmr_dir is not specified assume same location
    if not fmr_dir: fmr_dir = input_dir
    
    # open up the isovoxeled uniden image
    doc_vmr = bv.open_document(join(input_dir, vmr_fn))
    
    # do the coregistration
    doc_vmr.coregister_fmr_to_vmr_using_bbr(join(fmr_dir, fmr_fn))
    return(doc_vmr)
    

def copy_mesh(bv, input_dir, vmr_fn, masked_suffix='masked_'):
    """copy meshes for other filename, usefull if brainvoyager can use existing meshes (for example in bbr)
    bv just has to have the same dimensions, and is just uses so bv can actually load the meshes"""

    # set re search pattern 
    re_pattern = '(?=.*{})[a-zA-Z0-9._-]+.srf$'.format(masked_suffix)

    # search files
    dir_items = os.listdir(input_dir)
    dir_match = sorted([s for s in dir_items if re.search(re_pattern, s)])

    # open some vmr file so we can load mashes
    doc_vmr = bv.open_document(join(input_dir, vmr_fn))

    # loop over mesh files
    for mesh_fn in dir_match:
    
        # open mesh
        doc_vmr.load_mesh(mesh_fn)
        mesh = doc_vmr.current_mesh
    
        # new filename + save
        mesh_new_fn = re.sub(masked_suffix, '', mesh.file_name)
        mesh.save_as(mesh_new_fn)
    return

## HELPER FUNCTIONS

def _fmr_fn(input_dir, suffix):
    """get fmr filename without prefix, if multiple return first"""
    re_pattern = r'[a-zA-Z0-9_.]+{}$'.format(suffix)
    dir_items = os.listdir(input_dir)
    return([s for s in dir_items if re.search(re_pattern, s)][0])

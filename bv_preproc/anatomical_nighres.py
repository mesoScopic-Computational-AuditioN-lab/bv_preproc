"""Functions for preprocessing brainvoyager anatmical files using nighres
functions created by Jorie van Haren (2022), and tested on brainvoyaer version 22.2. 
for any help email jjg.vanharen@maastrichtuniversity.nl"""

# import things we need
import numpy as np
import os
import nighres
from os.path import join
from nilearn import plotting
import itertools

import bvbabel
import nibabel

from bv_preproc.utils import (print_f, prefix)

## HIGH LEVEL FUNCTIONS - LOOPING OVER MULTIPLE DATASETS

def removedura_all(input_dir, pps, sess, uniden_fn='uniden_IIHC', inv1_fn='INV1.v16', inv2_fn='INV2.v16', 
                           uni_fn='UNI.v16', t1_fn=None, bkg_dis=8.0, save_mask=False, bv=None):
    """calculate remove duramask for all particpants and session and apply to all saving vmrs and v16s
    input: input_dir : directory where v16 files are located
           uniden_fn : (default uniden_IIHC) the image to apply the mask to, can be any image with same size as mask
           inv1_fn : first inversion bv image (v16)
           inv2_fn : second inversion bv image (v16)
           uni_fn  : raw uni bv image (v16)
           t1_fn   : optional (default None), t1 v16 image for improved estimation
           bkg_dist : default 8.0, maximum distance within the mask for dura to be at
           save_mask : optional (default False), or nifti file to save mask file to e.g. 'dura_mask.nii.gz'
           bv : optional (default None), BrainVoyager"""
    for pp, ses in itertools.product(pps, sess):
        duramask = create_removedura_mask(join(input_dir, prefix(pp,ses)), inv1_fn=inv1_fn, inv2_fn=inv2_fn, 
                           uni_fn=uni_fn, t1_fn=t1_fn, bkg_dis=bkg_dis, save_mask=save_mask, bv=bv)
        apply_removedura_mask(join(input_dir, prefix(pp,ses)), duramask, uniden_fn=uniden_fn, bv=bv)
    return

    
## MAIN FUNCTIONS

def create_removedura_mask(input_dir, inv1_fn='INV1.v16', inv2_fn='INV2.v16', 
                           uni_fn='UNI.v16', t1_fn=None, bkg_dis=8.0, save_mask=False, bv=None):
    """Create an alpha mask (0 to 1) of brainmatter - duramatter probability.
    method uses nighres to first skullstip and then mp2rage dura estimation
    returns apha multi dim array mask
    input: input_dir : directory where v16 files are located
           inv1_fn : first inversion bv image (v16)
           inv2_fn : second inversion bv image (v16)
           uni_fn  : raw uni bv image (v16)
           t1_fn   : optional (default None), t1 v16 image for improved estimation
           bkg_dist : default 8.0, maximum distance within the mask for dura to be at
           save_mask : optional (default False), or nifti file to save mask file to e.g. 'dura_mask.nii.gz'
           bv : optional (default None), BrainVoyager method - only used for logging
    returns numpy array of alpha values"""
    
    # update user
    print_f('\nCalculating brainmask with Dura removed from {}, {}, {} with background distance set to {}\n' \
            ' Inside folder: {}\n Using t1: {}\n Saving mask file: {}'.format(inv1_fn, inv2_fn, uni_fn, bkg_dis, 
                                                                             input_dir, t1_fn, save_mask), bv=bv)

    # load our v16 (vmr) data
    inv1_head, inv1_img = bvbabel.v16.read_v16(join(input_dir, inv1_fn))
    inv2_head, inv2_img = bvbabel.v16.read_v16(join(input_dir, inv2_fn))
    uni_head, uni_img = bvbabel.v16.read_v16(join(input_dir, uni_fn))

    # prepair our data to play ball with nighres
    inv1_img_nb = nibabel.Nifti1Image(inv1_img, np.eye(4))
    inv2_img_nb = nibabel.Nifti1Image(inv2_img, np.eye(4))
    uni_img_nb = nibabel.Nifti1Image(uni_img, np.eye(4))

    # load t1 if wanted
    if t1_fn:
        t1_head, t1_img = bvbabel.v16.read_v16(join(input_dir, t1_fn))
        t1_img_nb = nibabel.Nifti1Image(t1_img, np.eye(4))
    else:
        t1_img_nb = None

    # use nighres to do skullstripping
    skullstripping_results = nighres.brain.mp2rage_skullstripping(
                                                    second_inversion=inv2_img_nb,
                                                    t1_weighted=uni_img_nb,
                                                    t1_map=t1_img_nb)

    # use nighres to do mp2rage dura estimation      
    removedura_results = nighres.brain.mp2rage_dura_estimation(
                                                    second_inversion=inv2_img_nb,
                                                    skullstrip_mask=skullstripping_results['brain_mask'],
                                                    background_distance=bkg_dis)

    # calculate new mask, use dura probability as -alpha
    new_mask = (1 - removedura_results['result'].get_fdata()) * skullstripping_results['brain_mask'].get_fdata()

    # save mask if desired (in nifti format)
    if save_mask: 
        new_mask_nb = nibabel.Nifti1Image(new_mask, np.eye(4))
        nibabel.save(new_mask_nb, join(input_dir, save_mask))
    return(new_mask)

def apply_removedura_mask(input_dir, mask_img, uniden_fn='uniden_IIHC', bv=None):
    """Apply the remove dura mask to an bv image (both v16 and vmr)
    input: input_dir : the input direcotry where the bv files are located
           mask_img  : the alpha mask to use to remove dura (must be 0 to 1 and np array)
           uniden_fn : (default uniden_IIHC) the image to apply the mask to, can be any image with same size as mask
           bv : BrainVoyager method, used for logging, and opening resulting file"""
    
    # update user
    print_f('\nApplying mask to remove dura from {} in {}'.format(uniden_fn, input_dir),bv=bv)
    
    # load files to apply mask to
    header, img = bvbabel.vmr.read_vmr(join(input_dir, '{}.vmr'.format(uniden_fn)))
    headerv16, imgv16 = bvbabel.v16.read_v16(join(input_dir, '{}.v16'.format(uniden_fn)))
    
    # multiply mask with images
    img = img * mask_img
    img = img.astype(np.uint8)
    imgv16 = imgv16 * mask_img
    imgv16 = imgv16.astype(np.uint16)
    
    # save the files in bv format
    bvbabel.vmr.write_vmr(join(input_dir, '{}_masked.vmr'.format(uniden_fn)), header, img)
    bvbabel.v16.write_v16(join(input_dir, '{}_masked.v16'.format(uniden_fn)), headerv16, imgv16)
    
    # return approriate
    if bv: bv.open_document(join(input_dir, '{}_masked.vmr'.format(uniden_fn)))
    return(imgv16)


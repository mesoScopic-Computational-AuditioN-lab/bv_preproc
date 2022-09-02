"""Functions for creating brainvoyager anatomical files and preprocessing anatomical data
functions created by Jorie van Haren (2022), and tested on brainvoyaer version 22.2. 
for any help email jjg.vanharen@maastrichtuniversity.nl"""

# import things we need
import numpy as np
import os
import itertools
import re
import pickle
import pandas as pd
from pydicom import dcmread
from os.path import join
import scipy.ndimage

import bvbabel
import nibabel

from bv_preproc.prepdicoms import anatomical_dir_information
from bv_preproc.utils import (print_f, prefix,
                              target_dir, preproc_filenames)


## CREATE FUNCTIONS

def create_vmrs(bv, input_dir, output_dir, pps, sess, key='KeysMp2rage', format='vmr'):
    """function to loop over participants and sessions, obtain anatomical dictionary from dicom file headers and
    create vmr (and v16) files by running create_vmr for all pp's for all runs
    input bv (for running BrainVoyager functions), input_dir (where renamed Dicoms are located),
    output_dir (parent directory to save vmrs), pps (list of participants), sess (list of sessions),
    key (optional, keyname for creating vmr from dicom, look in anatomical_dir_information for namings),
    foramt (default vmr)"""
    # loop over pp and ses for complete proc pipeline
    for pp, ses in itertools.product(pps, sess):
        # get dicoms header information
        anat_dict = anatomical_dir_information(join(input_dir, prefix(pp, ses)), bv=bv)
        create_vmr(bv, anat_dict, join(input_dir, prefix(pp,ses)), join(output_dir, prefix(pp,ses)), key=key, format=format)
    return


def create_vmr(bv, anat_dict, input_dir, output_dir, key='KeysMp2rage', format='vmr'):
    """allias function for brainvoyager vmr creation function
    input func_dict (dictonary with header info from `anatomical_dir_information`), first_file, 
    optional = key (default KeysMp2rage) : what keys to use to create vmr files (loaded from anat_dict)
    output_folder (where to save)"""

    # for multiple instances of file, loop over all
    for f in anat_dict[key].keys():
        # set parameters to load mosaic fmr file
        file           = anat_dict[key][f][0]
        first_file     = '{}/{}'.format(input_dir, file)

        # create actual file
        doc_vmr = bv.create_vmr_dicom(first_file)

        # chance name and save in correct location + delete old
        # yes.. this is stupid, but brainvoyager create vmr allows for way less compared to the fmr format - always named untitled
        doc_vmr.save_as('{}/{}.{}'.format(output_dir, f, format))

    return(doc_vmr)


## PROCESSING FUNCTIONS - HIGHER LEVEL (LOOPING OVER MUTLIPLE DATASETS)

def correct_inhomogeneities_all(bv, input_dir, pps, sess, doc_vmr_fn='uniden', 
                            actract_brain = True,       # bool whether to include skull stripping step
                            n_cycles = 8,               # number of itterations for fitting bias field
                            wm_tissue_range = 0.25,     # threshold to detect whether regions contain one or two tissue types
                            wm_intensity_thresh = 0.3,  # threshold to seperate wm from gm
                            polynom_order = 3):         # order of polynom to fit 3d field       
    """ correct inhomogenities using brainvoyagers extended function. 
    loop over all participants and sessions"""
    for pp, ses in itertools.product(pps, sess):
        correct_inhomogeneities(bv, join(input_dir, prefix(pp, ses)), doc_vmr_fn=doc_vmr_fn,
                            actract_brain = actract_brain,
                            n_cycles = n_cycles,
                            wm_tissue_range = wm_tissue_range,
                            wm_intensity_thresh = wm_intensity_thresh,
                            polynom_order = polynom_order)
    return
    
def isovoxel_all(bv, input_dir, pps, sess, doc_vmr_fn, 
             res = 0.4,           # target resolution
             framing_cube = 768,  # framing dimensions of output vmr data
             interpolation = 2,   # interpolation method (1:trilinear, 2:cubic spline interpolation, 3:sinc interpolation)
             output_suffix = '_ISO-'):
    """isovoxel data to desired resolution,
    loop over pps and sessions"""
    for pp, ses in itertools.product(pps, sess):
        isovoxel(bv, join(input_dir, prefix(pp, ses)), doc_vmr_fn,
                 res = res,
                 framing_cube = framing_cube,
                 interpolation = interpolation,
                 output_suffix = output_suffix)
    return
    
def apply_erosion_mask_all(input_dir, pps, sess, mask_fn='uniden_BrainMask', uniden_fn='uniden_IIHC', outmask_int=0, bv=None):
    """the intensity to apply to all the things outside the new mask,
    loop over all particippants and sessions inputted"""
    for pp, ses in itertools.product(pps, sess):
        apply_erosion_mask(join(input_dir, prefix(pp,ses)), mask_fn=mask_fn, uniden_fn=uniden_fn, outmask_int=outmask_int, bv=bv)
    return

def erode_mask_all(input_dir, pps, sess, mask_prefix='uniden', itterations=6, bv=None):
    """erode the mask created in the homogenity correction step
    the idea is to get rid of the peel (and probably some of the grey matter) but leaving wm in tact
    warning: erosion needed (itterations) might differ between particpants, might be advised to use 'erode_mask' directly"""
    for pp, ses in itertools.product(pps, sess):
        erode_mask(join(input_dir, prefix(pp,ses)), mask_prefix=mask_prefix, itterations=itterations, bv=bv)
    return

def mp2rage_genuniden_all(input_dir, pps, sess, chosen_factor, filenameUNI='UNI.v16', filenameINV1='INV1.v16', 
                      filenameINV2='INV2.v16', uniden_output_filename='uniden.v16', 
                        savev16=True, savevmr=True, bv=None):
    """function to take mp2rage files uni, inv1 and inv2 and given a chosen denoising factor
    returns denoised images. loops over all participants and sessions wanted"""
    for pp, ses in itertools.product(pps, sess):
        mp2rage_genuniden(join(input_dir, prefix(pp,ses)), chosen_factor, filenameUNI=filenameUNI, filenameINV1=filenameINV1,
                          filenameINV2=filenameINV2, uniden_output_filename=uniden_output_filename,
                          savev16=savev16, savevmr=savevmr, bv=bv)
    return
        

## PROCESSING FUNCTIONS

def correct_inhomogeneities(bv, input_dir, doc_vmr_fn='uniden', 
                            actract_brain = True,       # bool whether to include skull stripping step
                            n_cycles = 8,               # number of itterations for fitting bias field
                            wm_tissue_range = 0.25,     # threshold to detect whether regions contain one or two tissue types
                            wm_intensity_thresh = 0.3,  # threshold to seperate wm from gm
                            polynom_order = 3):         # order of polynom to fit 3d field       
    """correct inhomogenities using brainvoyagers extended function, 
    then open up adjusted volume"""

    # filepath lambda and load
    doc_vmr = bv.open_document(join(input_dir, '{}.vmr'.format(doc_vmr_fn)))

    # get new filename
    fn = '{}_IIHC.vmr'.format(re.search(r'.+(?=\.)',doc_vmr.file_name)[0])

    # do inhomogenitie correction
    print_f('\nRunning inhomogenitie correction for: {}'.format(doc_vmr.file_name), bv=bv)
    doc_vmr.correct_intensity_inhomogeneities_ext(actract_brain, n_cycles, wm_tissue_range, wm_intensity_thresh, polynom_order)
    doc_vmr = bv.open_document(join(input_dir, fn))
    return(doc_vmr) 

def isovoxel(bv, input_dir, doc_vmr_fn, 
             res = 0.4,           # target resolution
             framing_cube = 768,  # framing dimensions of output vmr data
             interpolation = 2,   # interpolation method (1:trilinear, 2:cubic spline interpolation, 3:sinc interpolation)
             output_suffix = '_ISO-'):
    """isovoxel data to desired resolution"""

    # filepath lambda and load
    doc_vmr = bv.open_document(join(input_dir, '{}.vmr'.format(doc_vmr_fn)))

    #isovoxel
    print_f('\nIsovoxel {} to {} (framing: {})'.format(doc_vmr_fn, res, framing_cube), bv=bv)
    doc_vmr.transform_to_isovoxel(res, framing_cube, interpolation, '{}{}{}.vmr'.format(doc_vmr_fn,output_suffix,res))
    return(bv.open_document('{}{}{}.vmr'.format(doc_vmr_fn,output_suffix,res)))   


def apply_erosion_mask(input_dir, mask_fn='uniden_BrainMask', uniden_fn='uniden_IIHC', outmask_int=0, bv=None):
    """the intensity to apply to all the things outside the new mask"""
    print_f('\nApplying erosion mask to {}.vmr, outmask opasity set to: {}'.format(uniden_fn, outmask_int), bv=bv)

    # load eroded masking
    _, mask = bvbabel.vmr.read_vmr(join(input_dir, '{}.vmr'.format(mask_fn)))

    # load the files to apply the soft mask to
    header, img = bvbabel.vmr.read_vmr(join(input_dir, '{}.vmr'.format(uniden_fn)))
    headerv16, imgv16 = bvbabel.v16.read_v16(join(input_dir, '{}.v16'.format(uniden_fn)))

    # set mask values
    mask[mask == 0] = outmask_int
    mask[mask == np.max(mask)] = 1
    
    # then for img, and v16 img apply intensity mask
    img = img * mask
    img = img.astype(np.uint8)
    imgv16 = imgv16 * mask
    imgv16 = imgv16.astype(np.uint16)

    # save the files in bv format
    bvbabel.vmr.write_vmr(join(input_dir, '{}_masked.vmr'.format(uniden_fn)), header, img)
    bvbabel.v16.write_v16(join(input_dir, '{}_masked.v16'.format(uniden_fn)), headerv16, imgv16)
    return(bv.open_document(join(input_dir, '{}_masked.vmr'.format(uniden_fn))))
    

def erode_mask(input_dir, mask_prefix='uniden', itterations=6, bv=None):
    """erode the mask created in the homogenity correction step
    the idea is to get rid of the peel (and probably some of the grey matter) but leaving wm in tact"""
    print_f('\nEroding brainmask edges of {}_BrainMask.vmr for {} itterations'.format(mask_prefix, 
                                                                                      itterations), bv=bv)
    
    # load mask
    maskhead, maskimg = bvbabel.vmr.read_vmr(join(input_dir, '{}_BrainMask.vmr'.format(mask_prefix)))

    # set mask to binary
    mask = np.zeros(maskimg.shape)
    mask[maskimg > 0] = 1

    # erode the mask in bool and put back in original masking array
    mask = scipy.ndimage.binary_erosion(mask, iterations=itterations).astype(np.uint8)
    maskimg[mask == 0] = 0
    bvbabel.vmr.write_vmr(join(input_dir, '{}_BrainMask.vmr'.format(mask_prefix)), maskhead, maskimg)
    return

def mp2rage_genuniden(input_dir, chosen_factor, filenameUNI='UNI.v16', filenameINV1='INV1.v16', 
                      filenameINV2='INV2.v16', uniden_output_filename='uniden.v16', 
                        savev16=True, savevmr=True, bv=None):
    """function to take mp2rage files uni, inv1 and inv2 and given a chosen denoising factor
    returns denoised images"""
    
    ############################################
    # Note: Python implemention of matlab code https://github.com/khanlab/mp2rage_genUniDen.git mp2rage_genUniDen.m
    # Date: 2019/09/25
    # Author: original author YingLi Lu, adopted by Jorie van Haren - translated for brainvoyager
    ############################################

    np.seterr(all='ignore')
    print_f('\nDenoising MP2Rage, using: {}, {}, and {}, output file: {}'.format(filenameUNI, 
                                                                                 filenameINV1, 
                                                                                 filenameINV2, 
                                                                                 uniden_output_filename), bv=bv)

    # load data
    header, mp2rage_img = bvbabel.v16.read_v16(join(input_dir, filenameUNI))
    _, inv1_img = bvbabel.v16.read_v16(join(input_dir, filenameINV1))
    _, inv2_img = bvbabel.v16.read_v16(join(input_dir, filenameINV2))

    # adjust dimensions for slight mismatches of phase data
    if inv1_img.shape != mp2rage_img.shape:
        tp_img = np.zeros(mp2rage_img.shape)
        tp_img[:inv1_img.shape[0], :inv1_img.shape[1], :inv1_img.shape[2]] = inv1_img
        inv1_img = tp_img
    if inv2_img.shape != mp2rage_img.shape:
        tp_img = np.zeros(mp2rage_img.shape)
        tp_img[:inv2_img.shape[0], :inv2_img.shape[1], :inv2_img.shape[2]] = inv2_img
        inv2_img = tp_img

    mp2rage_img = mp2rage_img.astype('float64')
    inv1_img = inv1_img.astype('float64')
    inv2_img = inv2_img.astype('float64')

    if mp2rage_img.min() >= 0 and mp2rage_img.max() >= 0.51:
       # converts MP2RAGE to -0.5 to 0.5 scale - assumes that it is getting only positive values
        mp2rage_img = (
            mp2rage_img - mp2rage_img.max()/2)/mp2rage_img.max()
        integerformat = 1
    else:
        integerformat = 0

    # computes correct INV1 dataset
    inv1_img = np.sign(mp2rage_img)*inv1_img # gives the correct polarity to INV1

    # because the MP2RAGE INV1 and INV2 is a sum of squares data, while the
    # MP2RAGEimg is a phase sensitive coil combination.. some more maths has to
    # be performed to get a better INV1 estimate which here is done by assuming
    # both INV2 is closer to a real phase sensitive combination

    # INV1pos=rootsquares_pos(-MP2RAGEimg.img,INV2img.img,-INV2img.img.^2.*MP2RAGEimg.img);
    inv1pos = rootsquares_pos(-mp2rage_img,
                              inv2_img, -inv2_img**2*mp2rage_img)
    inv1neg = rootsquares_neg(-mp2rage_img,
                              inv2_img, -inv2_img**2*mp2rage_img)

    inv1final = inv1_img

    inv1final[np.absolute(inv1_img-inv1pos) > np.absolute(inv1_img-inv1neg)
              ] = inv1neg[np.absolute(inv1_img-inv1pos) > np.absolute(inv1_img-inv1neg)]
    inv1final[np.absolute(inv1_img-inv1pos) <= np.absolute(inv1_img-inv1neg)
              ] = inv1pos[np.absolute(inv1_img-inv1pos) <= np.absolute(inv1_img-inv1neg)]

    # usually the multiplicative factor shouldn't be greater then 10, but that
    # is not the case when the image is bias field corrected, in which case the
    # noise estimated at the edge of the imagemight not be such a good measure

    multiplyingFactor = chosen_factor
    noiselevel = multiplyingFactor*np.mean(inv2_img[:, -11:, -11:])

    # run the actual denoising function
    mp2rage_imgRobustPhaseSensitive = mp2rage_robustfunc( inv1final, inv2_img, noiselevel**2)

    # set to interger format
    mp2rageimg_img = np.round(4095*(mp2rage_imgRobustPhaseSensitive+0.5))

    # Save image
    mp2rageimg_img = nibabel.casting.float_to_int(mp2rageimg_img,'int16');
    if savev16: bvbabel.v16.write_v16(join(input_dir, uniden_output_filename), header, mp2rageimg_img)
    if savevmr: 
        mp2rageimg_img = np.uint8(np.round(225*(mp2rage_imgRobustPhaseSensitive+0.5)))
        vmrheader, _ = bvbabel.vmr.read_vmr(join(input_dir, '{}.vmr'.format(re.search(r'.+(?=\.)',filenameUNI)[0])))
        bvbabel.vmr.write_vmr(join(input_dir, '{}.vmr'.format(re.search(r'.+(?=\.)',uniden_output_filename)[0])), vmrheader, mp2rageimg_img) 
    
    return(header, mp2rageimg_img) 

def mp2rage_robustfunc(INV1, INV2, beta):
    """adaptation of matlab robust denoise function"""
    return (np.conj(INV1)*INV2-beta)/(INV1**2+INV2**2+2*beta)

def rootsquares_pos(a, b, c):
    # matlab:rootsquares_pos=@(a, b, c)(-b+sqrt(b. ^ 2 - 4 * a.*c))./(2*a)
    return (-b+np.sqrt(b**2 - 4*a*c))/(2*a)

def rootsquares_neg(a, b, c):
    # matlab: rootsquares_neg = @(a, b, c)(-b-sqrt(b. ^ 2 - 4 * a.*c))./(2*a)
    return (-b-np.sqrt(b**2 - 4*a*c))/(2*a)


## FILE CONVERTION FUNCTIONS

def vmr_to_nifti(vmr_path, convert_nans=False, bv=None):
    """create nifti files from vmr files, optional input convert_nans (default False) will convert nans to 0"""
    print_f("\nConverting {} to nifti".format(vmr_path), bv=bv)
    _, vmr = bvbabel.vmr.read_vmr(vmr_path)
    if convert_nans: vmr = np.nan_to_num(vmr)
    img = nibabel.Nifti1Image(vmr, affine=np.eye(4))
    nibabel.save(img, '{}.nii.gz'.format(re.search(r'.+(?=\.)',vmr_path)[0]))
    return(img)

def v16_to_nifti(v16_path, convert_nans=False, bv=None):
    """create nifti files from vmr files, optional input convert_nans (default False) will convert nans to 0"""
    print_f("\nConverting {} to nifti".format(v16_path), bv=bv)
    _, v16 = bvbabel.v16.read_v16(v16_path)
    if convert_nans: vmr = np.nan_to_num(v16)
    img = nibabel.Nifti1Image(v16, affine=np.eye(4))
    nibabel.save(img, '{}.nii.gz'.format(re.search(r'.+(?=\.)',v16_path)[0]))
    return(img)
    
def v16_to_vmr(dir, v16fn, headerfn, bv=None):
    """convert v16 and downcast it to vmr (16 to 8 bit conv)
    input directory, v16 filename, and header to use for vmr"""
    print_f('\nDowncasting {} file to vmr'.format(v16fn), bv=bv)
    # load data - v16 and header to use
    _, img = bvbabel.v16.read_v16(join(dir, v16fn)) 
    header, _ = bvbabel.vmr.read_vmr(join(dir, headerfn)) 

    # convert file
    img = np.round((img.astype('float64') * 225) / 4095)
    img = np.uint8(img)

    # write vmr file
    bvbabel.vmr.write_vmr(join(dir, '{}.vmr'.format(re.search(r'.+(?=\.)',v16fn)[0])), header, img) 
    return(img)

## COORDINATE CONVERTION FUNCTIONS

def axes_nifti_bv(img):
    """take nifti axes and translate into bv format"""
    img_bv = img[::-1, ::-1, ::-1]
    img_bv = np.transpose(img_bv, (0, 2, 1))
    return(img_bv)
    
def axes_bv_nifti(img):
    """take bv axes and translate into nifti format"""
    img_nifti = np.transpose(img, (0, 2, 1))
    img_nifti = img_nifti[::-1, ::-1, ::-1]
    return(img_nifti)
    
def roi_bv_box(img, x1, y1, z1, x2, y2, z2):
    """sellect box of interest based on x, y, z bv coordinates
    returns matrix in nifti format of that area"""
    img = axes_nifti_bv(img)
    img_box = img[x1:x2, z1:z2, y1:y2]
    img_box = axes_bv_nifti(img_box)
    return(img_box)

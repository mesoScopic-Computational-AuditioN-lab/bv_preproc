"""Functions for creating brainvoyager functional files and preprocessing functional data
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

import bvbabel
import nibabel

from bv_preproc.prepdicoms import functional_dir_information
from bv_preproc.utils import (print_f, prefix,
                              target_dir, preproc_filenames)
from bv_preproc.functional import *

dir_path = os.path.dirname(os.path.realpath(__file__))

## HIGH LEVEL FUNCTIONS (LOOPIN OVER PARTICIPANTS AND SESSIONS

def topup_prepair_all(input_dir, output_dir, pps, sess,
                  pa_run = 1,    # what run to use for pa (first x runs)
                  nr_vols = 'AP', # nr of volumes to use for ap/pa, if AP count measured ap and use this, else give int
                  sbref=True,    # include sbref to list to convert
                  slicor=True,   # include slice scan correction naming for convertion
                  motcor=True,   # include motion correction naming for convertion
                  hpfil=True,    # include highpass filter naming for convertion
                  tpsmo=True,    # include temporal smoothing naming conv
                  motcor_appa = True,   # include motion correction for appa file
                  dimension_appa = 't', # appa dimension parameter for merge
                  cmdprefix='',         # cmd extra prefix (e.g. for wsl add 'wsl -e ')
                  print_cmd=False,      # instead of running cmd, print comment for copy paste
                  bv=None): 
    """Prepair bv files to be used for topup, including obtaining dicom information,
    loading and preprocessing ap / pa files, converting all files to nifti format,
    and merging ap and pa into appa. loop over all participants
    input: raw dicom directory (input_dir), preprocessed files location (output_dir),
           the participant number (pp), the session number (ses), and a series of optional
           parameters"""
    for pp, ses in itertools.product(pps, sess):
        topup_prepair(input_dir, output_dir, pp, ses,
                      pa_run=pa_run, nr_vols=nr_vols, sbref=sbref, slicor=slicor, 
                      motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, motcor_appa=motcor_appa, 
                      dimension_appa=dimension_appa, cmdprefix=cmdprefix, print_cmd=print_cmd, bv=bv)
    return

def topup_run_all(input_dir, pps, sess,
                  conf_fil_dir='preloaded', b0name='b0.cnf', acqname='acqparams.txt',
                  outname='topup_results', motcor_appa=True,   # include motion correction for appa file)
                  cmdprefix='', print_cmd=False, bv=None):
    """run fsl topup using a merged ap/pa file and config files (b0/acq) loop over all participants and sessions"""
    for pp, ses in itertools.product(pps, sess):
        outfile = _regex_topup_outfile(join(input_dir, prefix(pp, ses)), motcor=motcor_appa)
        topup_run(join(input_dir, prefix(pp,ses)), outfile, conf_fil_dir=conf_fil_dir,
                  b0name=b0name, acqname=acqname, outname=outname, cmdprefix=cmdprefix, print_cmd=print_cmd, bv=bv)
    return    

def topup_apply_all(input_dir, pps, sess, resname='topup_results', conf_fil_dir='preloaded', 
                    acqname='acqparams.txt', out_suffix='_TOPUP', 
                    inindex=1, method='jac', interp='spline',
                    sbref=True,     # include sbref to list apply
                    slicor = True,  # include slice scan correction naming conv
                    motcor = True,  # include motion correction naming conv
                    hpfil = True,   # include highpass filter naming conv
                    tpsmo = True,   # include temporal smoothing naming conv
                    cmdprefix='', print_cmd=False, bv=None):  
    """loop over directories (pps and sess) and apply topup to nii files given parameters and naming conv. of files"""
    for pp, ses in itertools.product(pps, sess):
        topup_apply_runs(join(input_dir, prefix(pp,ses)), resname=resname, conf_fil_dir=conf_fil_dir,
                         acqname=acqname, out_suffix=out_suffix, inindex=inindex, method=method, interp=interp,
                         sbref=sbref, slicor=slicor, motcor=motcor, hpfil=hpfil, 
                         tpsmo=tpsmo, cmdprefix=cmdprefix, print_cmd=print_cmd, bv=bv)
    return
    
def topup_tobv_all(input_dir, pps, sess, suffix='', sbref=True, # include sbref to list of to converts
                   slicor = True,  # include slice scan correction naming conv
                   motcor = True,  # include motion correction naming conv
                   hpfil = True,   # include highpass filter naming conv
                   tpsmo = True,   # include temporal smoothing naming conv)
                   bv=None): 
    """loop over directory (pps and sess) and transform topup'ed nii files back to fmr, 
    given parameters (optional with additional suffix"""
    for pp, ses in itertools.product(pps, sess):
        topup_convert_to_fmr_runs(join(input_dir, prefix(pp,ses)), suffix=suffix, 
                                  sbref=sbref, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, bv=bv)
    return


## TOPUP FUNCTIONS

def topup_prepair(input_dir, output_dir, pp, ses,
                  pa_run = 1,    # what run to use for pa (first x runs)
                  nr_vols = 'AP', # nr of volumes to use for ap/pa, if AP count measured ap and use this, else give int
                  sbref=True,    # include sbref to list to convert
                  slicor=True,   # include slice scan correction naming for convertion
                  motcor=True,   # include motion correction naming for convertion
                  hpfil=True,    # include highpass filter naming for convertion
                  tpsmo=True,    # include temporal smoothing naming conv
                  motcor_appa = True,   # include motion correction for appa file
                  motcor_ses = False,   # refference session, default false: same as ses
                  dimension_appa = 't', # appa dimension parameter for merge
                  cmdprefix='',         # cmd extra prefix (e.g. for wsl add 'wsl -e ')
                  print_cmd=False,      # instead of running cmd, print comment for copy paste
                  bv=None): 
    """Prepair bv files to be used for topup, including obtaining dicom information,
    loading and preprocessing ap / pa files, converting all files to nifti format,
    and merging ap and pa into appa
    input: raw dicom directory (input_dir), preprocessed files location (output_dir),
           the participant number (pp), the session number (ses), and a series of optional
           parameters"""
    
    # obtain functional dictionary
    func_dict = functional_dir_information(join(input_dir, prefix(pp, ses)), bv=bv)
    
    # load and preprocesses pa / ap
    doc_fmr = create_fmr_topup(bv, func_dict, 
                               join(input_dir, prefix(pp, ses)),
                               join(output_dir, prefix(pp, ses)), 'AP', nr_vols=nr_vols, pa_run=pa_run)
    doc_fmr_ap = preproc_topup(bv, doc_fmr, output_dir, pp, ses, motcor_ses=motcor_ses)
    doc_fmr = create_fmr_topup(bv, func_dict, 
                               join(input_dir, prefix(pp, ses)), 
                               join(output_dir, prefix(pp, ses)), 'PA', nr_vols=nr_vols, pa_run=pa_run)
    doc_fmr_pa = preproc_topup(bv, doc_fmr, output_dir, pp, ses, motcor_ses=motcor_ses)
    
    # convert bv files to nifti
    topup_convert_to_nifti_appa(join(output_dir, prefix(pp,ses)), motcor=motcor_appa, bv=bv)
    topup_convert_to_nifti_runs(join(output_dir, prefix(pp,ses)), sbref=sbref, slicor=slicor, 
                                motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, bv=bv) 
    
    # merge pa/ap together
    _, outfile = topup_merge_appa(join(output_dir, prefix(pp,ses)), motcor=motcor_appa, 
                                  dimension=dimension_appa, cmdprefix=cmdprefix, print_cmd=print_cmd, bv=bv)
    return


def topup_merge_appa(input_dir, motcor=True, dimension='t', output_type='NIFTI_GZ', cmdprefix='', print_cmd=False, bv=None):
    """convert ap and pa nifti files into one, sellect files based on regex,
    if motcor is true, find that instead"""

    # set regex for motcorrection when true or false
    if motcor: r_motcor = '='
    else: r_motcor = '!' 

    # set search pattern to obtain topup files
    ap_pattern = 'TOPUP(?{}.*3DMC)(?=.*_AP_)[a-zA-Z0-9_]+.nii.gz$'.format(r_motcor)
    pa_pattern = 'TOPUP(?{}.*3DMC)(?=.*_PA_)[a-zA-Z0-9_]+.nii.gz$'.format(r_motcor)
    # search directory for ap and pa files
    dir_items = os.listdir(input_dir)
    ap_fil = [s for s in dir_items if re.search(ap_pattern, s)][0]
    pa_fil = [s for s in dir_items if re.search(pa_pattern, s)][0]

    # create cmd command
    mergecmd = _topup_merge_cmd(join(input_dir, ap_fil), 
                                join(input_dir, pa_fil), 
                                join(input_dir, re.sub('AP', 'APPA', ap_fil)), 
                                dimension=dimension, output_type=output_type, cmdprefix=cmdprefix)
    print_f("\nRunning shell: '{}'".format(mergecmd), bv=bv)
    if print_cmd: print(mergecmd) 
    else: os.system(mergecmd)    # and run it
    return(mergecmd, re.sub('AP', 'APPA', ap_fil))


def topup_run(input_dir, filname, conf_fil_dir='preloaded', b0name='b0.cnf', 
              acqname = 'acqparams.txt', outname='topup_results', 
              cmdprefix='', print_cmd=False, bv=None):
    """run fsl topup using a merged ap/pa file and config files (b0/acq)"""  
    # use the preloaded configuration files if conf_fil_dir = 'preloaded'
    if conf_fil_dir == 'preloaded' : conf_fil_dir = join(dir_path, 'configs')
    # create cmd command for running topup
    topupcmd = _topup_run_cmd(join(input_dir, filname), 
                              join(conf_fil_dir, acqname), 
                              join(conf_fil_dir, b0name), 
                              join(input_dir, outname), cmdprefix=cmdprefix)
    print_f("\nRunning shell: '{}'\n -fsl topup can take long to process... please be patient".format(topupcmd), bv=bv)
    if print_cmd: print(topupcmd)
    else: os.system(topupcmd)
    return(topupcmd, outname)


def topup_apply_runs(input_dir, resname='topup_results', conf_fil_dir='preloaded', 
                     acqname='acqparams.txt', out_suffix='_TOPUP', 
                     inindex=1, method='jac', interp='spline',
                            sbref=True,     # include sbref to list apply
                            slicor = True,  # include slice scan correction naming conv
                            motcor = True,  # include motion correction naming conv
                            hpfil = True,   # include highpass filter naming conv
                            tpsmo = True,   # include temporal smoothing naming conv
                            cmdprefix='', print_cmd=False, bv=None):  
    """loop over directory and apply topup to nii files given parameters and naming conv. of files"""
    # get filenames for given parameters & convert to full path
    filenames = preproc_filenames(input_dir, sbref=2, slicor=slicor, motcor=motcor, 
                                  hpfil=hpfil, tpsmo=tpsmo, topup=False, dtype='nii.gz')
    # if desired add sbref file(s)
    if sbref:
        sbreffilename = preproc_filenames(input_dir, sbref=3, slicor=False, motcor=motcor, 
                                          hpfil=False, tpsmo=False, topup=False, dtype='nii.gz')
        filenames += sbreffilename
    # main loop
    for fl in filenames:
        topup_apply(fl, resname, input_dir, conf_fil_dir=conf_fil_dir, acqname=acqname, 
                    out_suffix=out_suffix, inindex=1, method=method, interp=interp,
                    cmdprefix=cmdprefix, print_cmd=print_cmd, bv=bv)
    return  


def topup_apply(filname, resname, input_dir, conf_fil_dir='preloaded', acqname='acqparams.txt', 
                out_suffix='_TOPUP', inindex=1, method='jac', interp='spline',
                cmdprefix='', print_cmd=False, bv=None): 
    """input filname, topup results name, directory, configuration directory, and optional name op acq doc & suffix
    to keep using the same name set out_suffix to False, lastly one could change topup apply options from defaults"""

    # use the preloaded configuration files if conf_fil_dir = 'preloaded'
    if conf_fil_dir == 'preloaded' : conf_fil_dir = join(dir_path, 'configs')
    
    # add suffix to path name (if outsuffix is not False)
    if out_suffix: outputname = _add_suffix(filname, out_suffix)
    else: outputname = filname
    
    # create cmd command for applying topup
    applycmd = _topup_apply_cmd(join(input_dir, filname), 
                                join(conf_fil_dir, acqname), 
                                join(input_dir, resname), 
                                join(input_dir, outputname), 
                                inindex=inindex, method=method, interp=interp, cmdprefix=cmdprefix)
    print_f("\nRunning shell: {}".format(applycmd), bv=bv)
    if print_cmd: print(applycmd)
    else: os.system(applycmd)
    return(applycmd)


def _topup_merge_cmd(ap_path, pa_path, outname, dimension='t', output_type='NIFTI_GZ', cmdprefix=''):
    """helper function to get fsl merge command"""
    return('{}fslmerge -{} "{}" "{}" "{}"'.format(cmdprefix, dimension, outname, ap_path, pa_path))

def _topup_run_cmd(appafil, acqpar, b0cnf, topup_results, cmdprefix=''):
    """helper function to get fsl topup command"""
    return('{}topup --imain="{}" --datain="{}" --config="{}" --out="{}"'.format(cmdprefix, appafil, acqpar, b0cnf, topup_results))

def _topup_apply_cmd(nii_path, acqpar, topup_results, outname, inindex=1, method='jac', interp='spline', cmdprefix=''):
#NOTE TO SELF: ADD " " OVER FILE PATHS FOR FILE PATHS WITH SPACES :)
    """helper function to get fsl apply topup command on nifti files"""
    return('{}applytopup -i "{}" -a "{}" --topup="{}" --inindex={} --method={} --interp={} --verbose --out="{}"'.format(cmdprefix, nii_path, acqpar, topup_results, inindex, method, interp, outname))

## CREATE & PREPROCESS FUNCTIONS

def create_fmr_topup(bv, func_dict, data_folder, target_folder, appa, skip_volumes_start=0, skip_volumes_end=0, nr_vols='AP', pa_run=1):
    """allias function for brainvoyager mosiaic fmr creation function - for topup use (5 volumes)
    input functional dict, data folder, target folder, the number of volumes to skip (from start) - skip_volumes_start, 
    the number of volumes to skip in the end - skip_volumes_end, how many volumes to take (if nr_vols == 'AP' we use the AP vol count),
    and which run to use as PA - pa_run"""

    # set parameters to load mosaic fmr file
    if   appa == 'AP': first_file     = '{}/{}'.format(data_folder, func_dict['KeysAP'][0])
    elif appa == 'PA': first_file     = '{}/{}'.format(data_folder, func_dict['KeysRunMag'][pa_run-1])

    # if number of volumes is AP, count number of AP volumes and use that for both PA an AP, else take the given number
    if nr_vols == 'AP':
        n_volumes      = func_dict[func_dict['KeysAP'][0]]['NrVolumes_Scanned'] - skip_volumes_end
    else:
        n_volumes      = nr_vols
        
    n_slices       = func_dict[func_dict['KeysAP'][0]]['NrSlices']
    fmr_stc_filename = '{}_TOPUP_{}'.format(func_dict[func_dict['KeysAP'][0]]['PatientName'], appa)
    big_endian     = func_dict[func_dict['KeysAP'][0]]['_isBigIndian']
    mosaic_rows    = func_dict[func_dict['KeysAP'][0]]['Rows']
    mosaic_cols    = func_dict[func_dict['KeysAP'][0]]['Columns']
    slice_rows     = func_dict[func_dict['KeysAP'][0]]['SliceRows']
    slice_cols     = func_dict[func_dict['KeysAP'][0]]['SliceColumns']
    bytes_per_pixel = 2
    
    # create actual file - ap & pa
    bv.create_mosaic_fmr(first_file, n_volumes, skip_volumes_start, 
                         False, n_slices, fmr_stc_filename, 
                         big_endian, mosaic_rows, mosaic_cols, 
                         slice_rows, slice_cols, bytes_per_pixel, target_folder)
    doc_fmr = bv.active_document                                                # activate current document
    return(doc_fmr)  


def preproc_topup(bv, doc_fmr, output_dir, pp, ses, motcor_ses=False):
    """do preproc step (only motion correction) for ap and pa - used for topup, uses paramaters set in preproc_run"""
    # do preprocessing
    doc_fmr = preproc_run(bv, doc_fmr, output_dir, pp, ses, b_sli=False, b_mot=True, b_hghp=False, b_temps=False, motcor_ses=motcor_ses)
    return(doc_fmr)


# FILE CONVERSION

def topup_convert_to_nifti_appa(input_dir, motcor=True, bv=None):
    """given a directory, and whether to sellect the motion corrected 
    file pressent in the directory (defualt=true). convert files to nifti (nii.gz) format
    nifti files are places in same directory as 'dir' and follow same naming"""

    # set regex for motcorrection when true or false
    if motcor: r_motcor = '='
    else: r_motcor = '!' 

    # set search pattern to obtain topup files
    ap_pattern = 'TOPUP_(?{}.*3DMC)(?=.*AP)[a-zA-Z0-9_]+.fmr$'.format(r_motcor)
    pa_pattern = 'TOPUP_(?{}.*3DMC)(?=.*PA)[a-zA-Z0-9_]+.fmr$'.format(r_motcor)
    # search directory for ap and pa files
    dir_items = os.listdir(input_dir)
    ap_fil = [s for s in dir_items if re.search(ap_pattern, s)][0]
    pa_fil = [s for s in dir_items if re.search(pa_pattern, s)][0]

    # load fmr using bvbabel
    ap_img = fmr_to_nifti(join(input_dir, ap_fil), bv=bv)
    pa_img = fmr_to_nifti(join(input_dir, pa_fil), bv=bv)
    return

def topup_convert_to_nifti_runs(input_dir, sbref=True, # include sbref to list to convert
                            slicor = True,  # include slice scan correction naming conv
                            motcor = True,  # include motion correction naming conv
                            hpfil = True,   # include highpass filter naming conv
                            tpsmo = True,   # include temporal smoothing naming conv)
                            bv=None):  
    """loop over directory and transform files into nifti format"""
    # get filenames for given parameters & convert to full path
    filenames = preproc_filenames(input_dir, sbref=2, slicor=slicor, motcor=motcor, hpfil=hpfil, tpsmo=tpsmo)
    # if desired add sbref file(s)
    if sbref:
        sbreffilename = preproc_filenames(input_dir, sbref=3, slicor=False, motcor=motcor, hpfil=False, tpsmo=False, topup=False)
        filenames += sbreffilename  
    # main loop
    for fl in filenames:
        img = fmr_to_nifti(join(input_dir, fl), convert_nans=True, bv=bv)
    return      

def topup_convert_to_fmr_runs(input_dir, suffix='', sbref=True, # include sbref to list of to converts
                              slicor = True,  # include slice scan correction naming conv
                              motcor = True,  # include motion correction naming conv
                              hpfil = True,   # include highpass filter naming conv
                              tpsmo = True,   # include temporal smoothing naming conv)
                              bv=None): 
    """loop over directory and transform topup'ed nii files back to fmr, given parameters (optional with additional suffix"""
    # get filenames for given parameters & convert to full path
    filenames = preproc_filenames(input_dir, sbref=2, slicor=slicor, 
                                  motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, 
                                  topup=True, dtype='nii.gz')
    hedrnames = preproc_filenames(input_dir, sbref=2, slicor=slicor, 
                                  motcor=motcor, hpfil=hpfil, tpsmo=tpsmo, 
                                  topup=False, dtype='fmr')
    # if desired add sbref file to list
    if sbref:
        sbreffilename = preproc_filenames(input_dir, sbref=3, slicor=False, motcor=motcor, 
                                          hpfil=False, tpsmo=False, 
                                          topup=True, dtype='nii.gz')
        filenames += sbreffilename
        sbrefhedrname = preproc_filenames(input_dir, sbref=3, slicor=False, motcor=motcor, 
                                          hpfil=False, tpsmo=False, 
                                          topup=False, dtype='fmr')
        hedrnames += sbrefhedrname
    # main loop
    for fl, hf in zip(filenames, hedrnames):
        nifti_to_fmr(join(input_dir, fl), join(input_dir, hf), bv=bv)
    return      

def fmr_to_nifti(fmr_path, convert_nans=False, bv=None):
    """create nifti files from fmr files, optional input convert_nans (default False) will convert nans to 0"""
    print_f("\nConverting {} to nifti".format(fmr_path), bv=bv)
    _, fmr = bvbabel.fmr.read_fmr(fmr_path)
    if convert_nans: fmr = np.nan_to_num(fmr)     # quickly translate nans to zeros
    img = nibabel.Nifti1Image(fmr, affine=np.eye(4))
    nibabel.save(img, '{}.nii.gz'.format(re.search(r'.+(?=.*.fmr$)',fmr_path)[0]))
    return(img)   

def nifti_to_fmr(nii_path, use_header_file, suffix='', bv=None):
    """create fmr file from nifti file, used for conversion back (i.e. for files where header info is known)
    input nii_path: path of the nifti, use_header_file: fmr file to copy header from (adjust prefix),
    surfix: set surfix naming for file saving 
    (not always desirable, especialy if naming has been done in nifti already, default empty string)
    file is saved in same directory as nii file was located"""
    print_f("\nConverting {} to fmr\n -using header: {}".format(nii_path, use_header_file), bv=bv)

    # savepath lambda to add filetype after path
    savepath = lambda t:  '{}{}.{}'.format(re.search(r'.+(?=.*.nii.gz$)',nii_path)[0], suffix, t)

    # load nifiti data from path & and get array data 
    niidata = nibabel.load(nii_path)
    niidata = niidata.get_fdata()
    if niidata.ndim == 3: niidata = niidata[:, :, :, np.newaxis]    # if we have only one volume

    # take original header file
    fmr_head, _ = bvbabel.fmr.read_fmr(use_header_file)

    # write the stc file
    bvbabel.stc.write_stc(savepath('stc'), niidata, data_type=fmr_head["DataType"])
    # write the associated fmr file
    _adjust_fmr_prefix(use_header_file, savepath('fmr'))
    return

## HELPER FUNCTIONS

def _adjust_fmr_prefix(use_header_file, newfmr):
    """copy an fmr header file and save it under an new name (add surfix)
    internally change prefix information"""
    # read the sellected header (fmr) file in full
    headerfil = open(use_header_file, 'r')
    full_header = headerfil.read()
    headerfil.close()
    # search for the prefix line and the proceding word after this, replaces it with the fmr title
    prefixtxt = re.search('(?<=Prefix:)\s+"([a-zA-Z0-9_]+)',full_header).group(1)
    prefixtxt_add = re.search(r'[a-zA-Z0-9_]+(?=.fmr)', newfmr)[0]
    # replace this word with title
    full_header_adjusted = re.sub(prefixtxt, prefixtxt_add, full_header)
    # create new fmr data document with corrected prefix
    new_headerfil = open(newfmr, 'w')
    new_headerfil.write(full_header_adjusted)
    new_headerfil.close()
    return

def _add_suffix(filname, out_suffix):
    doctype = re.search(r'(?=\.)[a-zA-Z.]+', filname)[0]
    docsuffix = '{}{}'.format(out_suffix, doctype) 
    return(re.sub(doctype, docsuffix, filname))

def _regex_topup_outfile(input_dir, motcor=True):
    """helper function to find outfile"""
    if motcor: motcor_re = '='
    else: motcor_re = '!'
    # make re pattern
    re_pattern = '(?=.*TOPUP)(?=.*APPA)(?{}.*3DM)(?=.*nii)[a-zA-Z0-9_.]+$'.format(motcor_re)
    # search dirctiory
    dir_items = os.listdir(input_dir)
    return([s for s in dir_items if re.search(re_pattern, s)][0])

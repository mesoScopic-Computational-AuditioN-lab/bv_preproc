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

from bv_preproc.prepdicoms import functional_dir_information
from bv_preproc.utils import (print_f, prefix,
                              target_dir, preproc_filenames,
                              _sort_human)


## CREATE FUNCTIONS

def create_fmrs(bv, input_dir, output_dir, pps, sess, n_runs, skip_volumes_end, skip_volumes_start=0, create_sbreff=True):
    """function to loop over participants and sessions, obtain functional dictionary from dicom file headers and 
    create fmr files by running create_fmr for all pp's for all runs
    input bv (for running BrainVoyager functions), input_dir (where renamed Dicoms are located), 
    output_dir (parent directory to save fmrs), n_runs (list of runs to loop over), pps (list of participants),
    sess (list of sessions), skip_volume_end (how many volumes to skip in the end, e.g. 5 for nordic noise scans),
    skip_volumes_start (default=0, how many volumes to skip),
    create_sbref (default=True, create fmr for sbreff)"""
    # loop over pp and ses for complete proc pipeline
    for pp, ses in itertools.product(pps, sess): 
        # get dicom header information
        func_dict = functional_dir_information(join(input_dir, prefix(pp, ses)), bv=bv)
        # start loop
        for currun in n_runs:
            # 1. create fmr documents
            create_fmr(bv, func_dict, 
                       join(input_dir, prefix(pp, ses)), 
                       join(output_dir, prefix(pp, ses)), 
                       currun, 
                       skip_volumes_end,
                       skip_volumes_start=skip_volumes_start)      # function creates fmr file for sellected run
        # 2. create sbreff document
        if create_sbreff: create_fmr_sbreff(bv, func_dict, 
                                            join(input_dir, prefix(pp, ses)), 
                                            join(output_dir, prefix(pp, ses)))   
    return


def create_fmr(bv, func_dict, data_folder, target_folder, currun, skip_volumes_end, skip_volumes_start=0):
    """allias function for brainvoyager mosiaic fmr creation function
    input func_dict (dictonary with header info from `functional_dir_information`), first_file (filename of first vol), 
    target_folder (where to save), currrun (current run),
    skip_volumes_end (how many valumes to skip in the end, e.g. 5 for nordic nois scans), 
    skip_volumes_start (default=0, how many volumes to skip)"""

    runidx = currun-1    # index for current run

    # set parameters to load mosaic fmr file
    first_file     = '{}/{}'.format(data_folder, func_dict['KeysRunMag'][runidx])

    n_volumes      = func_dict[func_dict['KeysRunMag'][runidx]]['NrVolumes_Scanned'] - skip_volumes_end
    first_volume_amr = False
    n_slices       = func_dict[func_dict['KeysRunMag'][runidx]]['NrSlices']
    fmr_stc_filename = '{}_run{}_FMR'.format(func_dict[func_dict['KeysRunMag'][runidx]]['PatientName'], currun)
    big_endian     = func_dict[func_dict['KeysRunMag'][runidx]]['_isBigIndian']
    mosaic_rows    = func_dict[func_dict['KeysRunMag'][runidx]]['Rows']
    mosaic_cols    = func_dict[func_dict['KeysRunMag'][runidx]]['Columns']
    slice_rows     = func_dict[func_dict['KeysRunMag'][runidx]]['SliceRows']
    slice_cols     = func_dict[func_dict['KeysRunMag'][runidx]]['SliceColumns']
    bytes_per_pixel = 2
    
    # create actual file
    bv.create_mosaic_fmr(first_file, n_volumes, skip_volumes_start, 
                         first_volume_amr, n_slices, fmr_stc_filename, 
                         big_endian, mosaic_rows, mosaic_cols, 
                         slice_rows, slice_cols, bytes_per_pixel, target_folder)
    return


def create_fmr_sbreff(bv, func_dict, data_folder, target_folder, currun=1):
    """allias function for brainvoyager mosiaic fmr creation function
    input func_dict (dictonary with header info from `functional_dir_information`), first_file (filename of first vol), 
    target_folder (where to save), currrun (current run),
    skip_volumes_end (how many valumes to skip in the end, e.g. 5 for nordic nois scans), 
    skip_volumes_start (default=0, how many volumes to skip)"""

    runidx = currun-1    # index for current run

    # set parameters to load mosaic fmr file
    first_file     = '{}/{}'.format(data_folder, func_dict['KeysRunReff'][runidx])
    fmr_stc_filename = '{}_run{}_SBRef_FMR'.format(func_dict[func_dict['KeysRunReff'][runidx]]['PatientName'], currun)
    first_volume_amr = False
    n_slices       = func_dict[func_dict['KeysRunReff'][runidx]]['NrSlices']
    big_endian     = func_dict[func_dict['KeysRunReff'][runidx]]['_isBigIndian']
    mosaic_rows    = func_dict[func_dict['KeysRunReff'][runidx]]['Rows']
    mosaic_cols    = func_dict[func_dict['KeysRunReff'][runidx]]['Columns']
    slice_rows     = func_dict[func_dict['KeysRunReff'][runidx]]['SliceRows']
    slice_cols     = func_dict[func_dict['KeysRunReff'][runidx]]['SliceColumns']
    bytes_per_pixel = 2

    # create actual file
    bv.create_mosaic_fmr(first_file, 1, 0, first_volume_amr, n_slices, fmr_stc_filename,         # here we do not create a amr image, 
                         big_endian, mosaic_rows, mosaic_cols,                        # we will do so after distorion correction
                         slice_rows, slice_cols, bytes_per_pixel, target_folder)
    return

## PREPROCESSING FUNCTIONS

def preprocess_fmrs(bv, output_dir, pps, sess, process_sbreff=True, b_sli=True, b_mot=True, b_hghp=True, b_temps=False,
                slictim_int_meth = 2,    # 0:linear, 1:cubic spline, 2: sinc
                motcor_full = True,      # motion correction uses full dataset or only subset of voxels for motion detection
                motcor_run = 1,          # what run to use as refference
                motcor_intp_meth = 2,    # 1:trilinear, 2: trilinear(motion det) sinc(final volume transform, 3: sinc,
                motcor_ses = False,      # refference session, default false: same as ses
                motcor_fromsdm = None,   # lambda function for obtaining sdm list - lambda pp, ses (default, motion correction from scratch)
                motcor_interses = 'interses_smooth_3DMC.sdm',  # -needed if fromsdm was given, fn within folder of interest (ignored otherwise)
                motcor_fromsdm_sbref = None, # lambda function for obtaining sdm list for sbref - lambda pp, ses (default ignored)
                highp_cycles = 7,        # number of dycles that will be removed from voxel time course   
                temps_fwhm = 2,          # float fwhm of gaussian smoothing kernel  
                temps_fwhm_s = 'TR',     # specify unit of fwhm (secs or some other timepoint)  
                fnlist = None,           # lambda function for obtaining fnlist - lambda pp, ses (default automatically loaded)
                fnsbref = None):         # lambda function for obtaining sbreflist - lambda pp, ses (default automatically loaded)
    """Do precocessing for all run (for doc_fmr) standard options / default options are set within function
    input doc_fmr and boleans whether to run preprocessing step
      b_sli: run slice time correction
      b_mot: run motion correction
      b_hghp: run highpass filter
      b_temps: run temporal smoothing
    by default all true except for temporal smoothing
    note that function only does brainvoyager function (topup has its own function)"""
    # loop over pp and ses for complete proc pipeline
    for pp, ses in itertools.product(pps, sess): 
        # check if we have a defined list
        if fnlist:
            fmrlist = fnlist(pp, ses)
        else:
            # load a list of to process files
            fmrlist = preproc_filenames(join(output_dir, prefix(pp, ses)),
                                    sbref = 2, slicor = False, motcor = False,
                                    hpfil = False, tpsmo = False, topup = False, dtype = 'fmr')
        # [optional] if we give in sdm lambda, motion parameters will be obtained from this instead of motion correction
        if motcor_fromsdm: sdmlist = motcor_fromsdm(pp, ses)
        else: sdmlist = [None] * len(fmrlist)
        # start loop over runs
        for i in range(len(fmrlist)):
            fmr = fmrlist[i]
            doc_fmr = bv.open_document(join(output_dir, prefix(pp, ses), fmr))
            doc_fmr = preproc_run(bv, doc_fmr, output_dir, pp, ses, b_sli=b_sli, b_mot=b_mot, 
                                  b_hghp=b_hghp, b_temps=b_temps, slictim_int_meth=slictim_int_meth,
                                  motcor_full=motcor_full, motcor_run=motcor_run, 
                                  motcor_intp_meth=motcor_intp_meth, motcor_ses=motcor_ses,
                                  motcor_fromsdm=sdmlist[i], motcor_interses=motcor_interses,
                                  highp_cycles=highp_cycles, temps_fwhm=temps_fwhm, temps_fwhm_s=temps_fwhm_s)
        # processes sbref if desired
        if process_sbreff:
            # check if we have a predefined sbref list
            if fnsbref: 
                sbrefflist = fnsbref(pp, ses)
            else:
                # load a list of to process files
                sbrefflist = preproc_filenames(join(output_dir, prefix(pp, ses)),
                                    sbref = 3, slicor = False, motcor = False,  
                                     hpfil = False, tpsmo = False, topup = False, dtype = 'fmr')
            # [optional] if we give in sdm lambda, motion parameters will be obtained from this instead of motion correction
            if motcor_fromsdm_sbref: sbreffsdmlist = motcor_fromsdm_sbref(pp, ses)
            else: sbreffsdmlist = [None] * len(sbrefflist)
            # start actual motion correction
            doc_fmr = bv.open_document(join(output_dir, prefix(pp, ses), sbrefflist[0]))
            doc_fmr = preproc_run_sbref(bv, doc_fmr, output_dir, pp, ses, motcor_ses=motcor_ses, 
                                        motcor_fromsdm=sbreffsdmlist[0], motcor_interses=motcor_interses)
    return

def preproc_run(bv, doc_fmr, output_dir, pp, ses, b_sli=True, b_mot=True, b_hghp=True, b_temps=False,
                slictim_int_meth = 2,    # 0:linear, 1:cubic spline, 2: sinc
                motcor_full = True,      # motion correction uses full dataset or only subset of voxels for motion detection
                motcor_run = 1,          # what run to use as refference
                motcor_intp_meth = 2,    # 1:trilinear, 2: trilinear(motion det) sinc(final volume transform, 3: sinc,
                motcor_ses = False,      # refference session, default false: same as ses
                motcor_fromsdm = None,   # instead apply motion correction from existing sdm file (fn within folder)
                motcor_interses = 'interses_smooth_3DMC.sdm',  # -needed if fromsdm was given, fn within folder of interest (ignored otherwise)
                highp_cycles = 7,        # number of dycles that will be removed from voxel time course   
                temps_fwhm = 2,          # float fwhm of gaussian smoothing kernel  
                temps_fwhm_s = 'TR'):    # specify unit of fwhm (secs or some other timepoint)  
    """Do precocessing for current run (for doc_fmr) standard options / default options are set within function
    input doc_fmr and boleans whether to run preprocessing step
      b_sli: run slice time correction
      b_mot: run motion correction
      b_hghp: run highpass filter
      b_temps: run temporal smoothing
    by default all true except for temporal smoothing
    note that function only does brainvoyager functions (topup has its own function)"""
    
    # do actual preprocessing
    if b_sli:   
        doc_fmr.correct_slicetiming_using_timingtable(2)           # do slicetime correction relying on sinc interpolation for interpolation 
        doc_fmr = bv.open(doc_fmr.preprocessed_fmr_name, True)     # open it to make it ready for next preproc step
    if b_mot:
        # check if a sdm_fn was inputted   
        if motcor_fromsdm: 
            if type(motcor_interses) == str: 
                trxyz = _parse_interses_sdm(join(output_dir, prefix(pp, ses), motcor_interses))  # if inter_ses_allign is string: assume sdm file
                # update user
                print_f('\nParsing through sdm file: {}\n -tx:{}, ty: {}, tz:{}, rx:{}, ry:{}, rz:{}'.format(join(prefix(pp, ses), motcor_interses),
                                                                                        trxyz[0],trxyz[1],trxyz[2],trxyz[3],trxyz[4],trxyz[5]), bv=bv)
            else: 
                trxyz = motcor_interses                            # else asume its an xyz rotation + rotation array
            tx, ty, tz, rx, ry, rz = trxyz                         # get motion translations and rotations from array
            doc_fmr.apply_motion_params(join(output_dir, prefix(pp, ses), motcor_fromsdm), 
                                        motcor_intp_meth, tx, ty, tz, rx, ry, rz)                # apply from existing file
            doc_fmr = bv.open(doc_fmr.preprocessed_fmr_name, True)
        else:
            if not motcor_ses: motcor_ses = ses
                                                                   # do motion correction to first volume of first run:   
            doc_fmr.correct_motion_to_run_ext(_path_fmr(output_dir, pp, motcor_ses, motcor_run),  # (targetfmr, targetvol, interp(trilinsinc),
                                            0, motcor_intp_meth, motcor_full, 150, False, True)   # full_data, maxitter, movie, extendedlog)   
            doc_fmr = bv.open(doc_fmr.preprocessed_fmr_name, True)
    if b_hghp:  
        doc_fmr.filter_temporal_highpass_glm_fourier(highp_cycles) # highpas filter results
        doc_fmr = bv.open(doc_fmr.preprocessed_fmr_name, True)
    if b_temps: 
        doc_fmr.smooth_temporal(temps_fwhm, temps_fwhm_s)          # temporal smooth results
        doc_fmr = bv.open(doc_fmr.preprocessed_fmr_name, True)
    return(doc_fmr)
    
def preproc_run_sbref(bv, doc_fmr, output_dir, pp, ses, motcor_ses=False, motcor_fromsdm=None, motcor_interses=None):
    """do preproc step (only motion correction) also for sbref, uses paramaters set in preproc_run
    note that function only does brainvoyager functions (topup has its own function)"""
    # do preprocessing
    doc_fmr = preproc_run(bv, doc_fmr, output_dir, pp, ses, b_sli=False, b_mot=True, b_hghp=False, b_temps=False, 
                          motcor_ses=motcor_ses, motcor_fromsdm=motcor_fromsdm, motcor_interses=motcor_interses)
    return(doc_fmr)

def interses_files_create(bv, output_dir, pps, sess, currun=1, out_fn='interses_smooth.fmr', meanovervols=False, 
                          b_sli=True, b_mot=True, b_hghp=False, b_temps=False, b_topup=False):
    """function to create intersession files, both smoothed (mean) volumes
    input output directory [output_dir], pp list [pps], ses list [sess], what run to smooth (default first) [currun], 
        output filename [out_fn], how many volumes to use in averaging (first x) - if False all volumes are used [meanovervols], 
        - what volume to sellect based on bv naming convention: b_sli, b_mot, b_hghp, b_temps, b_topup - True or False"""
    # loop over participants and sessions
    for pp, ses in itertools.product(pps, sess):

        # get by run sorted list of all fmr files of interest
        fmr_fn = _sort_human(preproc_filenames(join(output_dir, prefix(pp, ses)),
                             sbref = 2, slicor = b_sli, motcor = b_mot,
                             hpfil = b_hghp, tpsmo = b_temps, topup = b_topup, dtype = 'fmr'))

        # update user
        print_f('\nGenerating average volume of: {}\n -directory: {}\n -output filename: {}\n -mean over nr of volumes: {}'.format(fmr_fn[currun-1], 
                                                                                                                      join(output_dir, prefix(pp, ses)),
                                                                                                                      out_fn, meanovervols), bv=bv)
        # create mean value from parameters + what run we want
        _create_mean_fmr(join(output_dir, prefix(pp, ses), fmr_fn[currun-1]), 
                         join(output_dir, prefix(pp, ses), out_fn),
                         meanovervols=meanovervols)
    return

def interses_files_allign(bv, output_dir, pps, fromses, toses, fromallign_fn='interses_smooth.fmr', toallign_fn='interses_smooth.fmr'):
    """function to allign files between sessions, input: output directory [output_dir], pp list to parse [pps], 
    what session to allign [fromses], to session to allign [toses], allignment filenames [fromallign_fn & toallign_fn],
    by default both files have the same name (encoded by the directory)"""
    # loop over participants
    for pp in pps:
        # update user
        print_f('\nAllign {} (session:{}) to {} (session:{})'.format(fromallign_fn, prefix(pp,fromses), 
                                                                     toallign_fn, prefix(pp,toses)), bv=bv)

        # open document
        doc_fmr = bv.open_document(join(output_dir, prefix(pp, fromses), fromallign_fn))
        doc_fmr.correct_motion_to_run_ext(join(output_dir, prefix(pp, toses), toallign_fn),        # (targetfmr, targetvol, interp(trilinsinc),
                                          0, 3, True, 150, False, True)                           #  full_data, maxitter, movie, extendedlog)  
    return

## HELPER FUNCTIONS FOR PARSING SDMS AND CREATING MEAN VOLUMES
def _parse_interses_sdm(sdm_fn):
    """helper function to parse a sdm file and take rotation 
    and translation values from last line (only usefull for 1 volume sdms)"""
    # open sdm file and read xyz information for translations and rotations
    with open(sdm_fn) as f:
        trxyz = f.readlines()[-1]
    # transform to list of floats
    trxyz = [float(x) for x in re.findall(r'[\d.-]+', trxyz)]
    return(trxyz)

def _create_mean_fmr(fmr_fn, out_fn,meanovervols=False):
    """helper function to take fmr and create a mean fmr [:,:,:,1],
    optionally set number of volumes (from start) to use in mean, 
    if False use all"""
    # read file
    _, img = bvbabel.fmr.read_fmr(fmr_fn)
    # if meanovervols is false use all
    if not meanovervols: meanovervols = img.shape[-1]
    # calculate mean volume
    img_smooth = np.expand_dims(np.mean(img[:,:,:,:meanovervols], axis=-1), axis=-1)
    # save stc
    bvbabel.stc.write_stc('{}stc'.format(out_fn[:-3]), img_smooth)
    _adjust_fmr_volumes(fmr_fn, out_fn, nr_vols=1)
    return

def _adjust_fmr_volumes(use_header_file, newfmr, nr_vols=1):
    """copy an fmr header file and save it under an new name (add surfix)
    internally change prefix information and adjust the new number of volumes"""
    # read the sellected header (fmr) file in full
    headerfil = open(use_header_file, 'r')
    full_header = headerfil.read()
    headerfil.close()
    # search for the prefix line and the proceding word after this, replaces it with the fmr title
    prefixtxt = re.search('(?<=Prefix:)\s+"([a-zA-Z0-9_]+)',full_header).group(1)
    prefixtxt_add = re.search(r'[a-zA-Z0-9_]+(?=.fmr)', newfmr)[0]
    # replace this word with title
    full_header_adjusted = re.sub(prefixtxt, prefixtxt_add, full_header)
    # search for number of volumes line and proceding word after this, replaces it with new number of volumes
    voltxt = re.search('(?<=NrOfVolumes:)\s+([a-zA-Z0-9_]+)',full_header).group(1)
    full_header_adjusted = re.sub(voltxt, str(nr_vols), full_header_adjusted)
    # create new fmr data document with corrected prefix
    new_headerfil = open(newfmr, 'w')
    new_headerfil.write(full_header_adjusted)
    new_headerfil.close()
    return


## HELPER FUNCTIONS ##
def _path_sbref_amr(output_dir, pp, ses, currun):
    dir_loc = join(output_dir, prefix(pp, ses))
    dir_items = os.listdir(dir_loc)                                                                            # check documents in folder
    fmrdocs = list(filter(lambda v: re.search(r'run{}_SBRef[a-zA-Z_]+.amr$'.format(currun), v), dir_items))    # regex search for wanted
    return('{}/{}'.format(dir_loc, fmrdocs[0]))

def _path_sbref_fmr(output_dir, pp, ses, currun):
    dir_loc = join(output_dir, prefix(pp, ses))
    dir_items = os.listdir(dir_loc)                                                                            # check documents in folder
    fmrdocs = list(filter(lambda v: re.search(r'run{}_SBRef_FMR.fmr$'.format(currun), v), dir_items))    # regex search for wanted
    return('{}/{}'.format(dir_loc, fmrdocs[0]))

def _path_fmr(output_dir, pp, ses, currun):
    dir_loc = join(output_dir, prefix(pp, ses))
    dir_items = os.listdir(dir_loc)                                                                            # check documents in folder
    fmrdocs = list(filter(lambda v: re.search(r'run{}_FMR.fmr$'.format(currun), v), dir_items))    # regex search for wanted
    return('{}/{}'.format(dir_loc, fmrdocs[0]))

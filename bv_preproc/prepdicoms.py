"""Functions for parsing through Siemens dicom files and headers and make them ready for preprocessing.
not all functions within this file are necessary, the redundency is build in to facilitate parralell approaches
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

from bv_preproc.utils import (print_f, prefix,
                              target_dir)

## FUNCTIONS

def prep_dicoms(input_dir, pps, sess, bv):
    """Function to rename dicoms and clean up some of the header information.
    function is crusial for other more high level (bv) functions to function correctly 
    and have one common starting point. note that the underlaying bv.anonymize_dicoms is really slow
    so this function can take some time to complete, especially for many pp / ses.
    input:     input_dir =   parent directory where all pp folders of raw dicoms are located
               pps =         list of participants (or single pp)
               sess =        list of sessions (or single ses) 
               bv =          the brainvoyager python module"""
    # loop over pp and ses for complete proc pipeline
    for pp, ses in itertools.product(pps, sess): 
        bv.anonymize_dicoms(join(input_dir, prefix(pp, ses)), prefix(pp, ses))    # rename decoms
        
def prep_outputs(output_dir, pps, sess, bv=None):
    """Function to prepair output directory, only create when they do not yet exist
    input:     output_dir =     parent output direcotry where all pp folders will be located
               pps =         list of participants (or single pp)
               sess =        list of sessions (or single ses) 
    optional in: bv =        the brainvoyager python module """
    # loop over pp and ses for complete proc pipeline
    for pp, ses in itertools.product(pps, sess): 
        target_folder = target_dir(output_dir, prefix(pp,ses), bv) # find target path and create if needed

def functional_dir_information(input_dir, functional_marker='MOSAIC', save_vol_naming=False, bv=None):
    """scan files within sellected directory and abstract usefull information
    store files of interest together with important header information in dictonary and returns
    input: dir =     directory of interest (folder where all (brainvoyager namechanged) dicoms are located
    optional in:  functional_marker =    'MOSAIC' default - marker to gether wheter file in functional data
        save_vol_naming =    False default - save list of all namings for volumes and volume numbers [_volumes, _volumeNames]
    output:     returns dictonary nested by files of interest, includes keyfiles for catagories"""
    print_f('\nParsing through (functional) DICOM headers in "{}" - saving as dict'.format(input_dir), bv)
    # predefine dictonaries
    functional_dict = {}
    tempdict = {}

    # take note of what is in the directory and take first file of set
    dir_items = os.listdir(input_dir)
    dir_match = [s for s in dir_items if re.search(r'0+1-0+1.\w+$', s)]

    # loop over first files in directory
    for file in dir_match:

        # try to read dcmfiles (unless error in dcm run)
        try: curhead = dcmread(join(input_dir, file))        # read header files
        except: 
            curhead.ImageType[-1] = None        # set to none if error
            print_f('error reading dcm in {}'.format(file), bv)

        if curhead.ImageType[-1] == functional_marker:        # check if current file is functional data

            # regex the first part (session + run) of the filename to use later in getting volumes
            findrun = re.search(r'[a-zA-Z0-9]+-[0-9]+-', file)[0]         
            functional_dict[file] = {}                        # nest dictonary and save as filename of interest
            functional_dict[file]['FileName'] = file          # also save within for completeness
        
            # get information from header data and save
            functional_dict[file]['ImageMapping'] = curhead.ImageType[2].replace('P', 'Phase').replace('M', 'Magnitude')# phase or frequency
            # set run number
            if re.search(r'(?<=run)[0-9]+',curhead.SeriesDescription):
                functional_dict[file]['Run'] = int(re.search(r'(?<=run)[0-9]+',curhead.SeriesDescription)[0])
            else:
                functional_dict[file]['Run'] = np.nan
            functional_dict[file]['Rows'] = curhead.Rows
            functional_dict[file]['Columns'] = curhead.Columns
            functional_dict[file]['SliceRows'] = curhead.AcquisitionMatrix[0]
            functional_dict[file]['SliceColumns'] = curhead.AcquisitionMatrix[-1]
            functional_dict[file]['NrSlices'] = curhead[0x19, 0x100a].value
            functional_dict[file]['AcquisitionType'] = curhead.MRAcquisitionType

            # misc info
            functional_dict[file]['PatientName'] = curhead.PatientName
            functional_dict[file]['StudyDate'] = curhead.StudyDate
            functional_dict[file]['StudyTime'] = curhead.StudyTime    

            # other information
            functional_dict[file]['_description'] = curhead.SeriesDescription
            functional_dict[file]['_isPA'] = bool(re.search(r'_PA',curhead.SeriesDescription))
            functional_dict[file]['_isReff'] = bool(re.search(r'SBRef$',curhead.SeriesDescription)) # is file a refference image
            functional_dict[file]['_isBigIndian'] = not curhead.is_little_endian
            functional_dict[file]['_volumes'] = sorted([int(re.search(r'(?<={})(?=.*.dcm)[0-9]+'.format(findrun), 
                                                                      v)[0]) for v in os.listdir(input_dir) if 
                                                                      re.search(r'(?<={})(?=.*.dcm)[0-9]+'.format(findrun), v)])
            functional_dict[file]['_volumeNames'] = sorted(list(filter(lambda v: re.search(r'(?<={})(?=.*.dcm)[0-9]+'.format(findrun), 
                                                                                           v), os.listdir(input_dir))))

            # calculate number of volumes
            functional_dict[file]['NrVolumes_Scanned'] = functional_dict[file]['_volumes'][-1]    

            # delete vol naming info if not wanted
            if save_vol_naming == False: [functional_dict[file].pop(key, None) for key in ['_volumes', '_volumeNames']]

    # save keys per catagory for easy access
    tempdict['KeysAll'] = list(functional_dict.keys())
    tempdict['KeysRun'] = [r for r in functional_dict.keys() if functional_dict[r]['Run'] >0]
    tempdict['KeysRunMag'] = [r for r in functional_dict.keys() if (functional_dict[r]['Run'] >0) and 
                                          (functional_dict[r]['ImageMapping'] == 'Magnitude') and
                                          (functional_dict[r]['_isReff'] == False)]
    tempdict['KeysRunPhase'] = [r for r in functional_dict.keys() if (functional_dict[r]['Run'] >0) and 
                                          (functional_dict[r]['ImageMapping'] == 'Phase')]
    tempdict['KeysRunReff'] = [r for r in functional_dict.keys() if (functional_dict[r]['Run'] >0) and 
                                          (functional_dict[r]['_isReff'])]
    tempdict['KeysAP'] = [r for r in functional_dict.keys() if (functional_dict[r]['_isPA']) and
                                          (functional_dict[r]['ImageMapping'] == 'Magnitude') and
                                          (functional_dict[r]['_isReff'] == False)]
                                          
    # store in same dict
    functional_dict.update(tempdict)
    return(functional_dict)
    
def anatomical_dir_information(input_dir, anatomical_marker='3D', save_vol_naming=False, bv=None):
    """scan files within sellected directory and abstract usefull information
    store files of interest together with important header information in dictonary and returns
    input: dir =     directory of interest (folder where all (brainvoyager namechanged) dicoms are located
    optional in:  functional_marker =    'MOSAIC' default - marker to gether wheter file in functional data
        save_vol_naming =    False default - save list of all namings for volumes and volume numbers [_volumes, _volumeNames]
    output:     returns dictonary nested by files of interest, includes keyfiles for catagories"""

    print_f('\nParsing through (anatomical) DICOM headers in "{}" - saving as dict'.format(input_dir), bv)
    # predefine dictonaries
    anatomical_dict = {}
    tempdict = {}

    # take note of what is in the directory and take first file of set
    dir_items = os.listdir(input_dir)
    dir_match = [s for s in dir_items if re.search(r'0+1-0+1.\w+$', s)]

    # loop over first files in directory
    for file in dir_match:

        # try to read dcmfiles (unless error in dcm run)
        try: curhead = dcmread(join(input_dir, file))        # read header files
        except: 
            curhead.ImageType[-1] = None        # set to none if error
            print_f('error reading dcm in {}'.format(file), bv)

        if curhead.MRAcquisitionType == anatomical_marker: 
    
            # regex the first part (session + run) of the filename to use later in getting volumes
            findrun = re.search(r'[a-zA-Z0-9]+-[0-9]+-', file)[0]         
            anatomical_dict[file] = {}                        # nest dictonary and save as filename of interest
            anatomical_dict[file]['FileName'] = file          # also save within for completeness       

            # get information from header data and save
            anatomical_dict[file]['ImageMapping'] = curhead.ImageType[2].replace('P', 'Phase').replace('M', 'Magnitude')# phase or frequency 
            # set run number
            anatomical_dict[file]['Run'] = np.nan
            anatomical_dict[file]['Rows'] = curhead.Rows
            anatomical_dict[file]['Columns'] = curhead.Columns
            anatomical_dict[file]['SliceRows'] = curhead.AcquisitionMatrix[0]
            anatomical_dict[file]['SliceColumns'] = curhead.AcquisitionMatrix[-1]
            anatomical_dict[file]['AcquisitionType'] = curhead.MRAcquisitionType

            # misc info
            anatomical_dict[file]['PatientName'] = curhead.PatientName
            anatomical_dict[file]['StudyDate'] = curhead.StudyDate
            anatomical_dict[file]['StudyTime'] = curhead.StudyTime    

            # other information
            anatomical_dict[file]['_description'] = curhead.SeriesDescription
            anatomical_dict[file]['_isINV1'] = bool(re.search(r'(?=.*mp2rage)[\w.]+INV1', curhead.SeriesDescription))
            anatomical_dict[file]['_isINV2'] = bool(re.search(r'(?=.*mp2rage)[\w.]+INV2', curhead.SeriesDescription))
            anatomical_dict[file]['_isUNI'] = bool(re.search(r'(?=.*mp2rage)[\w.]+UNI', curhead.SeriesDescription))
            anatomical_dict[file]['_isT1'] = bool(re.search(r'(?=.*mp2rage)[\w.]+T1', curhead.SeriesDescription))
            anatomical_dict[file]['_isAngulated'] = bool(re.search(r'angulated', curhead.SeriesDescription))
            anatomical_dict[file]['_isBigIndian'] = not curhead.is_little_endian
            anatomical_dict[file]['_volumes'] = sorted([int(re.search(r'(?<={})(?=.*.dcm)[0-9]+'.format(findrun), 
                                                                      v)[0]) for v in os.listdir(input_dir) if 
                                                                             re.search(r'(?<={})(?=.*.dcm)[0-9]+'.format(findrun), v)])
            anatomical_dict[file]['_volumeNames'] = sorted(list(filter(lambda v: re.search(r'(?<={})(?=.*.dcm)[0-9]+'.format(findrun), 
                                                                                           v), os.listdir(input_dir))))

            # calculate number of volumes    
            anatomical_dict[file]['NrVolumes_Scanned'] = anatomical_dict[file]['_volumes'][-1]    
    
            # delete vol naming info if not wanted
            if save_vol_naming == False: [anatomical_dict[file].pop(key, None) for key in ['_volumes', '_volumeNames']]

    # save keys per catagory for easy access
    tempdict['KeysAll'] = list(anatomical_dict.keys())
    tempdict['KeysRunMag'] = [r for r in tempdict['KeysAll'] if (anatomical_dict[r]['ImageMapping'] == 'Magnitude')]
    tempdict['KeysRunPhase'] = [r for r in tempdict['KeysAll'] if (anatomical_dict[r]['ImageMapping'] == 'Phase')]
    inv1 = [r for r in tempdict['KeysAll'] if (anatomical_dict[r]['_isINV1']) and (anatomical_dict[r]['ImageMapping'] == 'Magnitude')]
    inv2 = [r for r in tempdict['KeysAll'] if (anatomical_dict[r]['_isINV2']) and (anatomical_dict[r]['ImageMapping'] == 'Magnitude')]
    uni = [r for r in tempdict['KeysAll'] if (anatomical_dict[r]['_isUNI']) and (anatomical_dict[r]['ImageMapping'] == 'Magnitude')] 
    tempdict['KeysMp2rage'] = {'INV1': inv1, 'INV2':inv2, 'UNI': uni}      # combination to store mp2rage files
    tempdict['KeysT1'] = {'T1':[r for r in tempdict['KeysAll'] if (anatomical_dict[r]['_isT1'])]}
    tempdict['KeysAngulated'] = {'Angulated':[r for r in tempdict['KeysAll'] if (anatomical_dict[r]['_isAngulated'])]}
                     
    # store in same dict
    anatomical_dict.update(tempdict)
    return(anatomical_dict)

def extract_dir_information(input_dir, save_format='csv', save_naming='dicom_namings', save_html=True, bv=None):
    """scan files within sellected directory and abstract usefull information to a dataframe
    dataframe contains rows of usefull files, and columns of important header and file information
    usefull to get a quick overview of all dicom files within directory
    input: dir =     directory of interest (folder where all (brainvoyager namechanged) dicoms are located
    optional in:  save_format =    'csv' default - format to save dataframe in, xlsx, csv, txt supported
                  save_namings =   'dicom_namings' default - save file under what name (or optionally nested path)
                  save_html =     True default - safe html dataframe
                  bv =            Brainvoyager bv function, here only needed for printing to log
    output:     retuns pandas dataframe with overview, as well as saving the dataframe within the input directory"""

    # print information
    print_f('\nExtracting information from {} and saving\n -saved as: {}\n -save html: {}'.format(input_dir, save_naming, save_html), bv)
    
    # predefine dictonary
    dct = {'FileName' : [], 
            '_description' : [], 
            'ImageMapping' : [], 
            'NrVolumes_Scanned' : [], 
            'NrSlices' : [], 
            'Run' : [], 
            'Rows' : [], 
            'Columns' : [], 
            'SliceRows' : [], 
            'SliceColumns' : [], 
            'PatientName' : [], 
            'StudyDate' : [], 
            'StudyTime' : []}

    # parse directory for function and anatomical
    func_dict = functional_dir_information(input_dir, bv=bv)
    anat_dict = anatomical_dir_information(input_dir, bv=bv)

    # loop over anatomical data
    for key in anat_dict['KeysAll']:
        for dk in dct.keys(): 
            try: dct[dk] += [anat_dict[key][dk]]
            except: dct[dk] += [np.nan]
    # loop over function data
    for key in func_dict['KeysAll']:
        for dk in dct.keys(): 
            try: dct[dk] += [func_dict[key][dk]]
            except: dct[dk] += [np.nan]

    # put all in dataframe and change some names
    df = pd.DataFrame(dct)
    df = df.rename(columns = {'_description' : 'Description', 'NrVolumes_Scanned':'Volumes', 'NrSlices':'Slices'})

    # check if we want to export html file
    if save_html: df.to_html('{}/{}.html'.format(input_dir, save_naming))

    # export dataframe
    if save_format == 'xlsx': df.to_excel('{}/{}.xlsx'.format(input_dir, save_naming))
    else: df.to_csv('{}/{}.{}'.format(input_dir, save_naming, save_format))
    return(df)

def clean_dicom_naming_struct(input_dir, pps, sess, dirformat='{}S{:02d}_SES{}', prefix='', bv=None):
    """ Function to clean up filenames of raw files (before Brainvoyager rename Dicoms)
    Input parent directory (where all test folders are located), and select folders of interst
    give (optionally) a prefix to all file names and change
    
    input: input_dir      = parent directory (where folders, e.g. 'S01_SES2' are located)
           pp             = a participant of interest, or a list of pp of interest
           ses            = a session, or a list of sessions of interest
    optional:  dirformat  = format of directory folder namings (default: e.g. S01_SES1 style)
           prefix         = prefix to be put before new name 
           bv             = brainvoyager bv function, here only needed for printing to log"""
    
    # check if pp and ses of interest is in list format (if not put in list)
    if not isinstance(pps, list): pps   = [pps]
    if not isinstance(sess, list): sess = [sess]
    
    # print information
    print_f('\nCleaning filenames of raw Dicom files in {} for:\n -pp:{}\n -ses:{}'.format(input_dir, pps, sess), bv)

    # loop over pps and over ses within pps
    for pp, ses in itertools.product(pps, sess):  
         
        # construct curdir
        curdir     = dirformat.format(input_dir, pp, ses)     

        # loop over files and change name
        for filename in os.listdir(curdir):
            # get the wanted strings
            newname = re.findall("SUB[0-9][^.]+|PP[0-9][^.]+|P[0-9][^.]+|S[0-9][^.]+|\.[0-9\.]+IMA", filename)

            # test empty string (for other possible files in map)
            if newname:
                # add personal prefix
                newname = ''.join([prefix] + newname)
                # change to consistant format  
                newname = re.sub(r'SUB|PP|P', 'S', newname)
                # change naming and remove old
                os.replace(join(curdir, filename), join(curdir, newname))
    return

"""Functions for plotting brainvoyager anatmical files and preprocessing anatomical data
functions created by Jorie van Haren (2022), and tested on brainvoyaer version 22.2. 
for any help email jjg.vanharen@maastrichtuniversity.nl"""

# import things we need
import numpy as np
import os
import re
import os
import scipy
import scipy.ndimage
from nilearn import plotting

from os.path import join
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import animation

import bvbabel
import nibabel

from bv_preproc.utils import (print_f, prefix,
                              target_dir, preproc_filenames,
                              _regex_preproc_filenames)


## PLOTTING FUNCTIONS

def plot_mp2rage_denois(input_dir, unifn='UNI.v16', unidenfn='uniden.v16', bv=None):
    """plot mp2rage before and after denoising, relies on plot_anats function
    input directory and optionally unifn (uni filename) and uniden (denoised uni filename)"""
    print_f('\nMP2Rage denoising results', bv=bv)
    ax, fig = plot_anats(input_dir, unifn, unidenfn, bv=bv)
    plt.suptitle('MP2Rage denoised vs raw-image', fontsize=22)
    return( ax, fig)


def plot_anats(input_dir, img1, img2, bv=None):   
    """plot two anatomicals side by side for comparison
    input: directory where v16 or vmr files are located and the two vmr / v16s,
    returns figure and ax, plot"""
    
    # inform on plotting params
    print_f('\nPlotting img {} and {}'.format(img1, img2), bv=bv)

    # already get the refference slice
    if img1[-3:] == 'vmr':
        _, img1_img = bvbabel.vmr.read_vmr(join(input_dir, img1))
    elif img1[-3:] == 'v16': 
        _, img1_img = bvbabel.v16.read_v16(join(input_dir, img1)) 
    if img2[-3:] == 'vmr':
        _, img2_img = bvbabel.vmr.read_vmr(join(input_dir, img2))
    elif img2[-3:] == 'v16':    
        _, img2_img = bvbabel.v16.read_v16(join(input_dir, img2))
    imgshape    = img1_img.shape
    
    # initalize figure
    fig, ax = plt.subplots(2, 
                           3,
                           figsize=(10, 10), gridspec_kw={'width_ratios': [imgshape[0], imgshape[0], imgshape[1]]})

    # plot the images
    ax[0, 0].imshow(np.rot90(img1_img[:,:,int(imgshape[2]/2)]), cmap='gray',  aspect='auto') 
    ax[1, 0].imshow(np.rot90(img2_img[:,:,int(imgshape[2]/2)]), cmap='gray',  aspect='auto')
    ax[0, 1].imshow(np.rot90(img1_img[:,int(imgshape[1]/2),:]), cmap='gray',  aspect='auto')
    ax[1, 1].imshow(np.rot90(img2_img[:,int(imgshape[1]/2),:]), cmap='gray',  aspect='auto')
    ax[0, 2].imshow(np.rot90(img1_img[int(imgshape[0]/2),:,:]), cmap='gray',  aspect='auto')
    ax[1, 2].imshow(np.rot90(img2_img[int(imgshape[0]/2),:,:]), cmap='gray',  aspect='auto')
    
    # set naming of views
    ax[0, 0].set_title('Transverse', fontsize=18)
    ax[0, 1].set_title('Coronal', fontsize=18)
    ax[0, 2].set_title('Sagital', fontsize=18)

    for row in range(ax.shape[0]):
        for colmn in range(ax.shape[1]):
            ax[row,colmn].axes.xaxis.set_visible(False)
            ax[row,colmn].axes.yaxis.set_visible(False)

    # adjust overall and shared plot parameters
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.suptitle('Comparison between image {} and {}'.format(img1, img2), fontsize=22)
    plt.tight_layout()
    return(ax, fig)

def plot_removedura_mask(input_dir, mask_img, img_fn, bv=None):
    """Plot dura removal mask on top of anatomical iamge of the same size
    input: input_dir : the input directory where the bv files are located
        mask_img     : the alpha mask to display
        img_fn       : the image to display as background (must be same size as mask, should be vmr)
        bv           : Brainvoyager method, used for logging only"""


    # update user
    print_f('\nPlotting dura removal mask on top of {}'.format(img_fn), bv=bv)    

    # load anatomical image
    _, img = bvbabel.vmr.read_vmr(join(input_dir, img_fn))
    
    # convert files to nibable
    mask_img_nb = nibabel.Nifti1Image(mask_img, np.eye(4))
    img_nb = nibabel.Nifti1Image(img, np.eye(4))

    # set up plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    # plot ROI using build nilearn plotting function
    plotting.plot_roi(mask_img_nb, img_nb, 
                    annotate=False, black_bg=True, draw_cross=False,
                    cmap='autumn', figure=fig, axes=ax)
                    
    # adjust overal plot parameters
    plt.suptitle('Dura removed brainmasked plotted on top of {}'.format(img_fn), fontsize=22)
    return(ax,fig)

import string, os, sys, subprocess, shutil, time

import numpy as np
import nibabel as nib

def fit_csd_mrtrix(input_dwi, input_bvals, input_bvecs, mask_img, output_dir, response_algo='tournier', fod_algo='csd'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dwi_mif = output_dir + '/tmp_dwi.mif'
    response_file = output_dir + 'response_function.txt'
    dwi_fod = output_dir + 'fod.mif'
    dwi_fod_nifti = output_dir + 'fod.nii.gz'
    
    os.system('mrconvert -fslgrad ' + input_bvecs + ' ' + input_bvals + ' ' + input_dwi + ' ' + dwi_mif)
    os.system('dwi2response ' + response_algo + ' ' + dwi_mif + ' ' + response_file)
    os.system('dwi2fod ' + fod_algo + ' ' + dwi_mif + ' ' + response_file + ' ' + dwi_fod + ' -mask ' + mask_img)
    os.system('mrconvert ' + dwi_fod + ' ' + dwi_fod_nifti)



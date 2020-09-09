import string, os, sys, subprocess, shutil, time
from glob import glob

import numpy as np
import nibabel as nib

from dipy.segment.mask import median_otsu
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io import read_bvals_bvecs
from dipy.io.bvectxt import reorient_vectors

def calculate_mean_img(input_img, output_img):
    img = nib.load(input_img)
    data = img.get_data()

    mean_data = np.mean(data, 3)
    mean_img = nib.Nifti1Image(mean_data.astype(np.float32), img.affine, img.header)
    mean_img.set_sform(img.get_sform())
    mean_img.set_qform(img.get_qform())
    nib.save(mean_img , output_img)

def create_target_img(input_img, output_img, index=0):
    img = nib.load(input_img)
    data = img.get_data()
    target_img = nib.Nifti1Image(data[:,:,:, index].astype(np.float32), img.affine, img.header)
    target_img.set_sform(img.get_sform())
    target_img.set_qform(img.get_qform())
    nib.save(target_img, output_img)

def n4_bias_correct(input_img, output_img):
    os.system('N4BiasFieldCorrection -d 3 -i ' + input_img + ' -o ' + output_img)

def bias_correct_mrtrix(input_img, input_mask, output_img, method='-ants', input_bval='', input_bvec='', nthreads='0'):
    command='dwibiascorrect -mask ' + input_mask + ' ' + method

    if input_bval != '' and input_bvec != '':
        command += ' -fslgrad ' + input_bvec + ' ' + input_bval + ' '

    command += ' -force -quiet -nthreads ' + nthreads + ' ' + input_img + ' ' + output_img
    os.system(command)

def denoise_mrtrix(input_dwi, output_dwi, output_noise='', nthreads='0'):
    #This function uses MRTRix function dwidenoise to remove noise from images
    if(output_noise != ''):
        os.system('dwidenoise ' + input_dwi + ' ' + output_dwi + ' -noise ' + output_noise + ' -nthreads ' + nthreads + ' -quiet -force')
    else:
        os.system('dwidenoise ' + input_dwi + ' ' + output_dwi + ' -quiet -force')

def mrdegibbs_mrtrix(input_dwi, output_dwi, nthreads='0'):
    #This function uses MRTRix to perform Gibbs ringing correction
    os.system('mrdegibbs ' + input_dwi + ' ' + output_dwi  + ' -nthreads ' + nthreads + ' -quiet -force')

def denoise_dipy(input_dwi, input_bval, input_bvec, mask_image, output_dwi):
    #This function uses nlmeans as part of dipy to remove noise from images
    img = nib.load(input_dwi)
    data = img.get_data()
    mask = nib.load(mask_image).get_data()
    aff = img.get_affine()
    sform = img.get_sform()
    qform = img.get_qform()

    bvals, bvecs = read_bvals_bvecs(input_bval, input_bvec)
    values = np.array(bvals)
    ii = np.where(values == bvals.min())[0]

    sigma = estimate_sigma(data)
    sigma = np.mean(sigma[ii])

    den = nlmeans(data,sigma=sigma, mask=mask)

    den_img = nib.Nifti1Image(den.astype(np.float32), aff, img.header)
    den_img.set_sform(sform)
    den_img.set_qform(qform)
    nib.save(den_img, output_dwi)

def correct_header_orientation(img_path, new_x, new_y, new_z):

    img = nib.load(img_path)
    sform = img.get_sform()
    qform = img.get_qform()
    
    new_sform = img.get_sform()
    new_qform = img.get_qform()
    
    if new_x == 'y':
        new_sform[0] = sform[1]
        new_qform[0] = qform[1]
    if new_x == '-y':
        new_sform[0] = -1.00*sform[1]
        new_qform[0] = -1.00*qform[1]
    if new_x == 'z':
        new_sform[0] = sform[2]
        new_qform[0] = qform[2]
    if new_x == '-z':
        new_sform[0] = -1.00*sform[2]
        new_qform[0] = -1.00*qform[2]

    if new_y == 'x':
        new_sform[1] = sform[0]
        new_qform[1] = qform[0]
    if new_y == '-x':
        new_sform[1] = -1.00*sform[0]
        new_qform[1] = -1.00*qform[0]
    if new_y == 'z':
        new_sform[1] = sform[2]
        new_qform[1] = qform[2]
    if new_y == '-z':
        new_sform[1] = -1.00*sform[2]
        new_qform[1] = -1.00*qform[2]

    if new_z == 'x':
        new_sform[2] = sform[0]
        new_qform[2] = qform[0]
    if new_z == '-x':
        new_sform[2] = -1.00*sform[0]
        new_qform[2] = -1.00*qform[0]
    if new_z == 'y':
        new_sform[2] = sform[1]
        new_qform[2] = qform[1]
    if new_z == '-y':
        new_sform[2] = -1.00*sform[1]
        new_qform[2] = -1.00*qform[1]
    
    out_img = img
    out_img.set_sform(new_sform)
    out_img.set_qform(new_qform)
    out_img.to_filename(img_path)

import string, os, sys, subprocess, shutil, time

#Neuroimaging Modules
import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu

def mask_dipy(input_dwi, output_mask, output_dwi=''):

    img = nib.load(input_dwi)
    data = img.get_data()
    masked_data, mask = median_otsu(data, 2,2)

    aff = img.get_affine()
    sform = img.get_sform()
    qform = img.get_qform()

    #Save these files
    masked_img = nib.Nifti1Image(masked_data.astype(np.float32), aff, img.header)
    mask_img = nib.Nifti1Image(mask.astype(np.float32), aff,  img.header)

    masked_img.set_sform(sform)
    masked_img.set_qform(qform)
    mask_img.set_sform(sform)
    mask_img.set_qform(qform)

    nib.save(mask_img, output_mask)

    if output_dwi != '':
        nib.save(masked_img, output_dwi)

def mask_skull_strip(input_dwi, output_mask, output_dwi=''):

    output_root, img = os.path.split(output_mask)

    tmpImg = output_root + '/tmp.nii.gz'
    tmpMask = output_root + '/tmp_mask.nii.gz'

    os.system('fslmaths ' + input_dwi + ' -Tmean ' + tmpImg)
    os.system('3dSkullStrip -input ' + tmpImg + ' -prefix ' + tmpMask)

    os.system('fslmaths ' + tmpMask + ' -bin ' + output_mask)

    if output_dwi != '':
        os.system('fslmaths ' + input_dwi + ' -mas ' + output_mask + ' ' + output_dwi)

    os.system('rm -rf ' + tmpImg)
    os.system('rm -rf ' + tmpMask)

def mask_bet(input_img, output_mask, output_img='', f_threshold='', clean_neck=''):

    output_root, img = os.path.split(output_mask)
    tmpImg = output_root + '/tmp.nii.gz'
    tmpMask = output_root + '/tmp_mask.nii.gz'

    os.system('fslmaths ' + input_img + ' -Tmean ' + tmpImg)

    for i in range(0,5):
        os.system('N4BiasFieldCorrection -d 3 -i ' + tmpImg + ' -o ' + tmpImg)

    cmd = 'bet ' + tmpImg + ' ' + tmpMask

    if f_threshold != '':
        cmd+= ' -f ' + f_threshold
    if clean_neck != '':
        cmd+= ' -B'

    os.system(cmd)
    os.system('fslmaths ' + tmpMask + ' -bin ' + output_mask)

    if output_img != '':
        os.system('fslmaths ' + input_img + ' -mas ' + output_mask + ' ' + output_img)

    os.system('rm -rf ' + tmpImg)
    os.system('rm -rf ' + tmpMask)

def mask_mrtrix(input_dwi, input_bval, input_bvec, output_mask, output_dwi=''):

    output_dir = os.path.dirname(output_mask)
    tmp_dwi = output_dir + '/tmp.dwi.mif'
    os.system('mrconvert -quiet -force -fslgrad '+ input_bvec + ' ' + input_bval + ' ' + input_dwi + ' ' + tmp_dwi )
    os.system('dwi2mask ' +  tmp_dwi + ' ' + output_mask + ' -quiet -force')

    if output_dwi != '':
        os.system('fslmaths ' + input_dwi + ' -mas ' + output_mask + ' ' + output_dwi)

    os.system('rm -rf ' + output_dir + '/tmp*')

def mask_ants(input_img, output_mask, ref_img, ref_mask, output_img='', nthreads=1):

    output_root, img = os.path.split(output_mask)

    ants_output = output_root + '/tmp_ants_'
    tmp_img = output_root + '/tmp.nii.gz'
    tmp_mask = output_root + '/tmp_mask.nii.gz'

    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)

    os.system('fslmaths ' + input_img + ' -Tmean ' + tmp_img)
    os.system('antsBrainExtraction.sh -d 3 -o ' + ants_output + ' -a ' + tmp_img + ' -e ' + ref_img + ' -m ' + ref_mask)

    os.system('mv ' + ants_output + 'BrainExtractionMask.nii.gz ' + output_mask)
    os.system('fslmaths ' + output_mask + ' -bin ' + output_mask)

    if output_img != '':
        os.system('mv ' + ants_output + 'BrainExtractionBrain.nii.gz ' + output_img)

    os.system('rm -rf ' + output_root + '/tmp*')

def apply_mask(input_img, output_img, mask_img):
    os.system('fslmaths ' + input_img + ' -mas ' + mask_img + ' ' + output_img)

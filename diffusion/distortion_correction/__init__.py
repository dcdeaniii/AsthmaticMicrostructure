import string, os, sys, subprocess, shutil, time
from glob import glob

import numpy as np
import nibabel as nib
import utils.tools as img_tools

def topup_fsl(input_dwi, input_bvals, input_index, input_acqparams, output_topup_base, config_file='', field_output=''):

    #First, find the indices of the B0 images
    dwi_img = nib.load(input_dwi)
    aff = dwi_img.get_affine()
    sform = dwi_img.get_sform()
    qform = dwi_img.get_qform()
    dwi_data = dwi_img.get_data()

    bvals = np.loadtxt(input_bvals)
    index = np.loadtxt(input_index)
    acqparams = np.loadtxt(input_acqparams)
    ii = np.where(bvals == 0)

    b0_data = dwi_data[:,:,:,np.asarray(ii).flatten()]
    b0_indices = index[ii].astype(int)
    b0_acqparams=acqparams[b0_indices-1]

    output_dir = os.path.dirname(output_topup_base)
    tmp_acqparams = output_dir + '/tmp.acqparams.txt'
    tmp_b0 = output_dir + '/tmp.B0.nii.gz'

    b0_imgs = nib.Nifti1Image(b0_data, aff, dwi_img.header)
    nib.save(b0_imgs, tmp_b0)
    np.savetxt(tmp_acqparams, b0_acqparams, fmt='%.8f')

    topup_command='topup --imain='+tmp_b0+' --datain='+tmp_acqparams+' --out='+output_topup_base

    if config_file != '':
        topup_command += ' --config='+config_file
    if field_output != '':
        topup_command += ' --fout='+field_output

    print(topup_command)
    os.system(topup_command)
    os.system('rm -rf ' + output_dir + '/tmp*')

def epi_reg_fsl(input_dwi, input_bval, fieldmap, fieldmap_ref, struct_img, struct_brain, output_dwi, pedir, dwellTime, fm_ref_brain=''):

    dwi_img = nib.load(input_dwi)
    dwi_data = dwi_img.get_data()
    bvals = np.loadtxt(input_bval)
    ii = np.where(bvals == 0)

    output_dir = os.path.dirname(output_dwi)
    epi_ref = output_dir + '/tmp.epi.nii.gz'
    os.system('fslroi ' + input_dwi + ' ' + epi_ref + ' 0 1')

    #Align the structual image to the mean B0 to keep things in DWI spaces
    struct_img_aligned = output_dir + '/tmp.struct.nii.gz'
    struct_brain_aligned = output_dir + '/tmp.struct_brain.nii.gz'
    struct_img_mat = output_dir +'/tmp.struct_2_dwi.mat'
    os.system('flirt -in ' + struct_img + ' -ref ' + epi_ref + ' -out ' + struct_img_aligned + ' -omat ' + struct_img_mat + ' -searchrx -180 180 -searchrz -180 180 -searchry -180 180')
    os.system('flirt -in ' + struct_brain + ' -ref ' + epi_ref + ' -out ' + struct_brain_aligned + ' -applyxfm -init ' + struct_img_mat)

    bias_fieldmap_ref = output_dir + '/tmp.fm_ref.bias.nii.gz'
    os.system('N4BiasFieldCorrection -d 3 -i ' + fieldmap_ref + ' -o ' + bias_fieldmap_ref)
    if fm_ref_brain == '':
        struct2ref = output_dir + '/tmp.struct2ref.nii.gz'
        fm_ref_brain = output_dir + '/tmp.fm_ref_brain.nii.gz'
        fm_ref_omat = output_dir + '/tmp.fm_ref_brain.mat'
        os.system('flirt -in ' + struct_img + ' -ref ' +bias_fieldmap_ref + ' -omat ' + fm_ref_omat + ' -searchrx -180 180 -searchry -180 180 -searchrz -180 180')
        os.system('flirt -in ' + struct_brain + ' -ref ' +bias_fieldmap_ref + ' -out ' + struct2ref + ' -applyxfm -init ' + fm_ref_omat )
        os.system('fslmaths ' + struct2ref + ' -bin -fillh ' + struct2ref)
        os.system('fslmaths ' + fieldmap_ref + ' -mas ' + struct2ref + ' ' + fm_ref_brain)

    fm_rads = output_dir + '/tmp.fm.rads.nii.gz'
    os.system('fslmaths ' + fieldmap + ' -mul 6.28 ' + fm_rads)

    epi_reg_out = output_dir + '/tmp.epi_reg'
    os.system('epi_reg --epi=' + epi_ref + ' --t1=' + struct_img_aligned + ' --t1brain='+ struct_brain_aligned + ' --fmap=' + fm_rads + ' --fmapmag='+bias_fieldmap_ref+ ' --fmapmagbrain='+fm_ref_brain + ' --pedir=' + pedir + ' --echospacing='+dwellTime + ' --out='+epi_reg_out)

    #Apply warp to dwi series
    os.system('applywarp -i ' + input_dwi + ' -r ' + struct_img_aligned + ' -o ' + output_dwi + ' -w ' + epi_reg_out + '_warp.nii.gz --interp=spline --rel')
    os.system('rm -rf ' + output_dir + '/tmp*')

def fugue_fsl(input_dwi, input_bvals, input_fm, input_fm_ref, output_dwi, fieldmap_dir, unwarpdir, dwellTime, fm_ref_mask_img=''):

    if not os.path.exists(fieldmap_dir):
        os.mkdir(fieldmap_dir)

    if input_fm_ref.endswith('.nii'):
        input_fm_ref_base = input_fm_ref[0:len(input_fm_ref)-4]
    else:
        input_fm_ref_base = input_fm_ref[0:len(input_fm_ref)-7]

    if input_fm.endswith('.nii'):
        input_fm_base = input_fm[0:len(input_fm)-4]
    else:
        input_fm_base = input_fm[0:len(input_fm)-7]

    #Skull-strip the reference
    mask_img=''
    fm_ref_mask = fieldmap_dir + 'mask.nii.gz'
    if fm_ref_mask_img != '':
        mask_img = fm_ref_mask_img
    else:
        mask_img = fm_ref_mask
        os.system('N4BiasFieldCorrection -d 3 -i ' + input_fm_ref + ' -o ' + mask_img)
        os.system('bet ' + mask_img + ' ' + mask_img)
        os.system('fslmaths ' + mask_img + ' -bin -fillh -dilM -dilM -ero -ero -bin ' + mask_img)

    os.system('fslmaths ' + input_fm_ref + ' -mas ' + mask_img + ' ' + fm_ref_mask)

    fm_rads = fieldmap_dir + 'fmap_Radians.nii.gz'

    #Now scale the field map and mask
    os.system('fslmaths ' + input_fm + ' -mul 6.28 -mas ' + mask_img + ' ' + fm_rads)
    os.system('fugue --loadfmap='+fm_rads+' --despike -smooth 2 --savefmap='+fm_rads)

    input_fm_ref_warp = fieldmap_dir + 'fmap_warp.nii.gz'

    #Warp the reference image
    #os.system('fugue -i ' + fm_ref_mask + ' --unwarpdir='+unwarpdir + ' --dwell='+dwellTime + ' --loadfmap='+fm_rads + ' -w ' + input_fm_ref_warp)
    os.system('cp -r ' + fm_ref_mask + ' ' + input_fm_ref_warp)

    dwi_ref = fieldmap_dir + '/dwi_ref.nii.gz'
    bvals = np.loadtxt(input_bvals)
    ii = np.where(bvals != 0)

    dwi_img = nib.load(input_dwi)
    aff = dwi_img.get_affine()
    sform = dwi_img.get_sform()
    qform = dwi_img.get_qform()
    dwi_data = dwi_img.get_data()

    dwi_mean = np.mean(dwi_data, axis=3)
    dwi_mean_img = nib.Nifti1Image(dwi_mean, aff, dwi_img.header)
    nib.save(dwi_mean_img, dwi_ref)
    os.system('N4BiasFieldCorrection -d 3 -i ' + dwi_ref + ' -o ' + dwi_ref)
    os.system('bet ' + dwi_ref + ' ' + dwi_ref)

    #Align warped reference to the dwi data
    fm_ref_warp_align = fieldmap_dir + 'fmap_warp-aligned.nii.gz'
    fm_ref_mat = fieldmap_dir + 'fmap2dwi.mat'
    os.system('flirt -in ' + input_fm_ref_warp + ' -ref ' + dwi_ref + ' -out ' + fm_ref_warp_align + ' -omat ' + fm_ref_mat + ' -dof 6 -cost normmi')

    #Apply this to the field map
    fm_rads_warp = fieldmap_dir + 'fmap_Radians-warp.nii.gz'
    os.system('flirt -in ' + fm_rads + ' -ref ' + dwi_ref + ' -applyxfm -init ' + fm_ref_mat + ' -out ' + fm_rads_warp)

    #Now, undistort the image
    os.system('fugue -i ' + input_dwi + ' --icorr --unwarpdir='+unwarpdir + ' --dwell='+dwellTime + ' --loadfmap='+fm_rads_warp+' -u ' + output_dwi)


def prep_external_fieldmap(input_dwi, input_fm, input_fm_ref, dwellTime, unwarpdir, field_map_dir):

    if not os.path.exists(field_map_dir):
        os.mkdir(field_map_dir)

    #Skull-strip the reference
    if input_fm_ref.endswith('.nii'):
        input_fm_ref_base = input_fm_ref[0:len(input_fm_ref)-4]
    else:
        input_fm_ref_base = input_fm_ref[0:len(input_fm_ref)-7]

    fm_ref_mask = input_fm_ref_base + '.mask.nii.gz'

    os.system('bet ' + input_fm_ref + ' ' + fm_ref_mask)

    if input_fm.endswith('.nii'):
        input_fm_base = input_fm[0:len(input_fm)-4]
    else:
        input_fm_base = input_fm[0:len(input_fm)-7]

    fm_rads = input_fm_base + '.rads.nii.gz'

    #Now scale the field map and mask
    os.system('fslmaths ' + input_fm + ' -mul 6.28 -mas ' + fm_ref_mask + ' ' + fm_rads)

    input_fm_ref_warp = input_fm_ref_base + '.warp.nii.gz'
    #Warp the reference image
    os.system('fugue -i ' + fm_ref_mask + ' --unwarpdir='+unwarpdir + ' --dwell='+dwellTime + ' --loadfmap='+fm_rads + ' -w ' + input_fm_ref_warp)

    dwi_ref = field_map_dir + '/dwi_ref.nii.gz'
    os.system('fslroi ' + input_dwi + ' ' + dwi_ref + ' 0 1' )

    #Align warped reference to the dwi data
    fm_ref_warp_align = input_fm_ref_base + '.warp.aligned.nii.gz'
    fm_ref_mat = input_fm_ref_base + '_2_dwi.mat'
    os.system('flirt -in ' + input_fm_ref_warp + ' -ref ' + dwi_ref + ' -out ' + fm_ref_warp_align + ' -omat ' + fm_ref_mat)

    #Apply this to the field map
    fm_rads_warp = input_fm_base + '.rads.warp.nii.gz'
    os.system('flirt -in ' + fm_rads + ' -ref ' + dwi_ref + ' -applyxfm -init ' + fm_ref_mat + ' -out ' + fm_rads_warp)

    fm_hz_warp = input_fm_base + '.hz.warp.nii.gz'
    os.system('fslmaths ' + fm_rads_warp + ' -mul 0.1592 ' + fm_hz_warp)

def buddi_method(input_dwi, input_bvals, input_acqparams, output_dwi, input_t1w='', input_t2w='', threads='1'):

    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = threads

    output_dir, img = os.path.split(output_dwi)

    #First, find the indices of the B0 and DWI images
    dwi_img = nib.load(input_dwi)
    dwi_data = dwi_img.get_data()

    bvals = np.loadtxt(input_bvals)
    ii = np.where(bvals == 0)
    jj = np.where(bvals != 0)

    mean_b0 = output_dir + '/tmp_mean_b0.nii.gz'
    mean_b0_data = np.mean(dwi_data[:,:,:,np.asarray(ii).flatten()], 3)
    mean_b0_img = nib.Nifti1Image(mean_b0_data.astype(np.float32), dwi_img.get_affine(), dwi_img.header)
    mean_b0_img.set_sform(dwi_img.get_sform())
    mean_b0_img.set_qform(dwi_img.get_qform())
    nib.save(mean_b0_img, mean_b0)

    mean_dwi = output_dir + '/tmp_mean_dwi.nii.gz'
    mean_dwi_data = np.mean(dwi_data[:,:,:,np.asarray(jj).flatten()], 3)
    mean_dwi_img = nib.Nifti1Image(mean_dwi_data.astype(np.float32), dwi_img.get_affine(), dwi_img.header)
    mean_dwi_img.set_sform(dwi_img.get_sform())
    mean_dwi_img.set_qform(dwi_img.get_qform())
    nib.save(mean_dwi_img , mean_dwi)


    #Now bias correct the mean B0 and DWI
    img_tools.n4_bias_correct(mean_b0, mean_b0)
    img_tools.n4_bias_correct(mean_dwi, mean_dwi)

    #Bias correct structural image(s)
    tmp_t1 = ''
    tmp_t1_aligned = ''
    tmp_t2=''
    tmp_t2_aligned = ''
    tmp_t1_rigid_cmd = ''
    tmp_t2_rigid_cmd = ''

    ants_rigid_cmd = 'antsRegistration -d 3 -o ' + output_dir + '/tmp_ants_rigid_'

    if input_t1w != '':
        tmp_t1 = output_dir + '/tmp_t1w.nii.gz'
        tmp_t1_aligned = output_dir + '/tmp_t1w_dwiAligned.nii.gz'

        img_tools.n4_bias_correct(input_t1w, tmp_t1)
        ants_rigid_cmd += ' --metric Mattes['+mean_dwi+','+tmp_t1+',1,32,Regular,0.25]'


    if input_t2w != '':
        tmp_t2 = output_dir + '/tmp_t2w.nii.gz'
        tmp_t2_aligned = output_dir + '/tmp_t2w_dwiAligned.nii.gz'

        img_tools.n4_bias_correct(input_t2w, tmp_t2)
        ants_rigid_cmd += ' -m Mattes['+mean_dwi+','+tmp_t2+',1,32,Regular,0.25]'

    ants_rigid_cmd+=' --transform Rigid[0.1] --convergence [100x50x25,1e-4,10] --shrink-factors 8x4x2 --smoothing-sigmas 4x2x1vox'
    os.system(ants_rigid_cmd)

    if input_t1w != '':
        os.system('antsApplyTransforms -d 3 -i ' + tmp_t1 + ' -r ' + mean_dwi + ' -o ' + tmp_t1_aligned + ' -t ' + output_dir + '/tmp_ants_rigid_0GenericAffine.mat')
    if input_t2w != '':
        os.system('antsApplyTransforms -d 3 -i ' + tmp_t2 + ' -r ' + mean_dwi + ' -o ' + tmp_t2_aligned + ' -t ' + output_dir + '/tmp_ants_rigid_0GenericAffine.mat')

    #Nonlinear registration of DWI data to structural data, only in the phase encode direction
    #Determine the phase encode direction by reading the acquisition parameters
    acqparams = np.loadtxt(input_acqparams)
    ants_nonlinear_cmd = 'antsRegistration -d 3 -o ' + output_dir + '/tmp_ants_nonlinear_'

    if input_t1w != '':
        ants_nonlinear_cmd += ' --metric MI['+tmp_t1_aligned+','+mean_dwi+',1,32,Regular,0.25]'

    if input_t2w != '':
        ants_nonlinear_cmd += ' --metric MI['+tmp_t2_aligned+','+mean_dwi+',1,32,Regular,0.25]'

    ants_nonlinear_cmd += ' --convergence [100x50x25x10,1e-6,5] -t SyN[0.1,3,0] -f 8x4x2x1 -s 4x2x1x0mm -u 1 -z 1'

    #print(ants_nonlinear_cmd)
    os.system(ants_nonlinear_cmd)

    os.system('antsApplyTransforms -d 3 -e 3 -i ' + input_dwi + ' -r ' + mean_dwi + ' -o ' + output_dwi + ' -t ' + output_dir + '/tmp_ants_nonlinear_0Warp.nii.gz')
    os.system('rm -rf ' + output_dir + '/tmp*')

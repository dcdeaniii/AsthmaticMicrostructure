#!/home/dean/Desktop/Software/LINUX/Conda/bin/python

import os,sys, shutil
import Library.utils.tools as datacorr
import Library.diffusion.distortion_correction as distcorr
import Library.diffusion.eddy_correction as eddycorr
import Library.utils.masking as mask
import Library.diffusion.model_fitting.dti as dtifit
import Library.diffusion.model_fitting.noddi as noddifit
import Library.diffusion.utils as diff_util

study_dir = '/study/dean_k99/Studies/panisa/'
processingCode_dir = study_dir + 'processing-code/'
processedDataDirectory = study_dir + 'processed-data/'

###DEFINE STUDY SPECIFIC PARAMETERS###
study_bvals = processingCode_dir + 'bvals.bval'
study_bvecs = processingCode_dir + 'bvecs.bvec'
study_acqparams = processingCode_dir + 'acqparams.txt'
study_index = processingCode_dir + 'index.txt'
###

input = sys.argv[1:]
subject_id = input[0]
nThreads=str(input[1])

#AVAILABLE OPTIONS FOR DTI/FWE-DTI/DKI ALGORITHMS
#dti_fit_method: WLS, LS/OLS, NLLS, RESTORE
#fwe_fit_method: WLS, NLLS
#dki_fit_method: WLS, OLS

dti_fit_method = 'WLS'

#Define directories and image paths
subj_proc_path = processedDataDirectory + subject_id + '/dwi/'
rawdata_dir = subj_proc_path+'/rawdata/'
preprocess_dir = subj_proc_path+'/preprocessed/'
dti_results_dir = subj_proc_path+'/DTI/'
amico_results_dir = subj_proc_path+'/AMICO-NODDI/'
noddi_results_dir = subj_proc_path+'/NODDI-WATSON/'

if not os.path.exists(preprocess_dir):
    os.mkdir(preprocess_dir)

##################################
##################################
##### PROCESSING STARTS HERE #####
##################################
##################################

#First, make sure the number of dwis, bvals, and bvecs are consistent
if not os.path.exists(preprocess_dir + 'bvals.bval'):
    print('Checking DWI acquisition')
    diff_util.check_dwi_acquisition_params(input_dwi=rawdata_dir + 'dwi.nii.gz',
                                           input_bval=study_bvals,
                                           input_bvec=study_bvecs,
                                           input_index=study_index,
                                           output_bval=preprocess_dir + 'bvals.bval',
                                           output_bvec=preprocess_dir + 'bvecs.bvec',
                                           output_index=preprocess_dir + 'index.txt')


#Remove noise and then correct for Gibbs ringing
if not os.path.exists(preprocess_dir + 'dwi.denoise.nii.gz'):
    print('Performing Noise Correction...')
    datacorr.denoise_mrtrix(input_dwi=rawdata_dir + 'dwi.nii.gz',
                            output_dwi=preprocess_dir + 'dwi.denoise.nii.gz',
                            output_noise=preprocess_dir + 'noise.map.nii.gz',
                            nthreads=nThreads)

if not os.path.exists(preprocess_dir + 'dwi.denoise.gibbs.nii.gz'):
    print('Performing Gibbs Ringing Correction...')
    datacorr.mrdegibbs_mrtrix(input_dwi=preprocess_dir + 'dwi.denoise.nii.gz',
                              output_dwi=preprocess_dir + 'dwi.denoise.gibbs.nii.gz',
                              nthreads=nThreads)
#Create Mask
if not os.path.exists(preprocess_dir + 'mask.nii.gz'):
    print('Masking data...')
    mask.mask_bet(input_dwi=preprocess_dir + 'dwi.denoise.gibbs.nii.gz',
                  output_mask=preprocess_dir + 'mask.nii.gz')
#Run EDDY correction
if not os.path.exists(preprocess_dir + 'dwi.denoise.gibbs.eddy.nii.gz'):
    print('Running EDDY...')
    eddycorr.eddy_fsl(input_dwi=preprocess_dir + 'dwi.denoise.gibbs.nii.gz',
             input_bval=preprocess_dir + 'bvals.bval',
             input_bvec=preprocess_dir + 'bvecs.bvec',
             input_index=preprocess_dir + 'index.txt',
             input_acqparam=study_acqparams,
             mask_img=preprocess_dir + 'mask.nii.gz',
             output_dwi=preprocess_dir + 'dwi.denoise.gibbs.eddy.nii.gz',
             output_bvec=preprocess_dir + 'bvecs_eddy.rotated.bvecs',
             repol=1,
             data_shelled=1,
             nthreads=nThreads)


#Remove outliers determined by EDDY and threshold of data to remove:
if not os.path.exists(preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.nii.gz'):
    print('Running Outlier Correction...')
    diff_util.remove_outlier_imgs(input_dwi=preprocess_dir + 'dwi.denoise.gibbs.eddy.nii.gz',
                                  input_bval=preprocess_dir + 'bvals.bval',
                                  input_bvec=preprocess_dir + 'bvecs_eddy.rotated.bvecs',
                                  input_index=preprocess_dir + 'index.txt',
                                  input_acqparams=study_acqparams,
                                  input_mask=preprocess_dir + 'mask.nii.gz',
                                  eddy_output_basename=preprocess_dir + 'dwi.denoise.gibbs.eddy',
                                  output_dwi=preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.nii.gz',
                                  output_bval=preprocess_dir + 'corr_bvals.bval',
                                  output_bvec=preprocess_dir + 'corr_bvecs.bvec',
                                  output_index=preprocess_dir + 'corr_index.txt',
                                  output_removed_imgs_dir=preprocess_dir + '/QC',
                                  method='EDDY_QUAD')

#Run Fieldmap correction with fugue
if not os.path.exists(preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.fugue.nii.gz'):
    print('Running Distortion Correction with FUGUE...')
    distcorr.fugue_fsl(input_dwi=preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.nii.gz',
                       input_bvals=preprocess_dir + 'corr_bvals.bval',
                       input_fm=rawdata_dir + 'fieldmap.nii.gz',
                       input_fm_ref=rawdata_dir + 'fieldmap_ref.nii.gz',
                       output_dwi=preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.fugue.nii.gz',
                       field_map_dir=preprocess_dir,
                       unwarpdir='x',
                       dwellTime='0.000342')

#Bias Correct DWI data
if not os.path.exists(preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.fugue.bias.nii.gz'):
    print('Performing Bias-Field Correction...')
    datacorr.bias_correct_mrtrix(input_img=preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.fugue.nii.gz',
                                 input_mask=preprocess_dir + 'mask.nii.gz',
                                 output_img=preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.fugue.bias.nii.gz',
                                 input_bval=preprocess_dir + 'corr_bvals.bval',
                                 input_bvec=preprocess_dir + 'corr_bvecs.bvec')

#Update Mask after FUGUE Correction
print('Updating Masking data...')
mask.mask_bet(input_dwi=preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.fugue.bias.nii.gz',
              output_mask=preprocess_dir + 'mask.nii.gz')


if not os.path.exists(dti_results_dir + 'dti_FA.nii.gz'):
    print('Fitting tensor model with ' + dti_fit_method + '...')
    dtifit.fit_dti_dipy(input_dwi=preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.fugue.bias.nii.gz',
                        input_bval=preprocess_dir + 'corr_bvals.bval',
                        input_bvec=preprocess_dir + 'corr_bvecs.bvec',
                        output_dir=dti_results_dir,
                        fit_type=dti_fit_method,
                        mask=preprocess_dir + 'mask.nii.gz')


if not os.path.exists(amico_results_dir + 'noddi_FICVF.nii.gz'):
    print('Fitting AMICO-NODDI model...')
    noddifit.fit_noddi_amico(input_dwi=preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.fugue.bias.nii.gz',
                             input_bval=preprocess_dir + 'corr_bvals.bval',
                             input_bvec=preprocess_dir + 'corr_bvecs.bvec',
                             output_dir=amico_results_dir,
                             input_mask=preprocess_dir + 'mask.nii.gz')

if not os.path.exists(noddi_results_dir + 'noddi_ICVF.nii.gz'):
    print('Fitting NODDI model...')
    noddifit.fit_noddi_dmipy(input_dwi=preprocess_dir + 'dwi.denoise.gibbs.eddy.corr.fugue.bias.nii.gz',
                             input_bval=preprocess_dir + 'corr_bvals.bval',
                             input_bvec=preprocess_dir + 'corr_bvecs.bvec',
                             output_dir=noddi_results_dir,
                             input_mask=preprocess_dir + 'mask.nii.gz')


if not os.path.exists(subj_proc_path + 'PSEUDO-T1/pseudoT1.nii.gz'):
    print('Creating Pseudo T1-weighted Image')
    diff_util.create_pseudoT1_img(fa_img=dti_results_dir + 'dti_FA.nii.gz',
                                  fiso_img=noddi_results_dir+'noddi_ISO.nii.gz',
                                  mask_img=preprocess_dir + 'mask.nii.gz',
                                  pseudoT1_img=subj_proc_path + 'PSEUDO-T1/pseudoT1.nii.gz')

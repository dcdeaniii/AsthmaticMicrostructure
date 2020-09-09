import string, os, sys, subprocess, shutil, time
import numpy as np

if sys.platform == 'linux':
    eddy='eddy_openmp'
    eddy_cuda='eddy_cuda9.1'
else:
    eddy='eddy'

def eddy_correct_fsl(input_dwi, input_bvec, output_dwi, output_bvec, output_log):

    eddy_output_basename = output_dwi[0:len(output_dwi)-7]
    logFile = eddy_output_basename + '.ecclog'

    if os.path.exists(logFile):
        os.remove(logFile)

    command = 'eddy_correct ' + input_dwi + ' ' + eddy_output_basename + ' 0'
    os.system(command)

    os.system('mv ' + logFile + ' ' + output_log)

    #Rotate b-vecs after doing the eddy correction
    os.system('fdt_rotate_bvecs ' + input_bvec+ ' ' + output_bvec + ' ' + output_log)

def eddy_fsl(input_dwi, input_bval, input_bvec, input_index, input_acqparam, output_dwi, output_bvec, topup_base='', external_b0='', repol=0, data_shelled=0, mb='', cuda=False, mporder=0, ol_type='sw', mb_off='1', estimate_move_by_suscept=False, slice_order='', mask_img='', cuda_device=0, nthreads='1'):

    output_dir = os.path.dirname(output_dwi)
    tmp_mask = output_dir + '/tmp_mask.nii.gz'

    if mask_img == '':
        tmp_dwi = output_dir + '/tmp_img.nii.gz'
        os.system('fslroi ' + input_dwi + ' ' + tmp_dwi + ' 0 1')
        os.system('bet ' + tmp_dwi + ' ' + output_dir + '/tmp -m')
    else:
        os.system('cp ' + mask_img + ' ' + tmp_mask)

    eddy_output_basename = output_dwi[0:len(output_dwi)-7]
    if cuda:
        command = 'CUDA_VISIBLE_DEVICES='+str(cuda_device)+ ' ' + eddy_cuda + ' --imain=' + input_dwi + ' --mask=' + tmp_mask + ' --index=' + input_index + ' --acqp=' + input_acqparam + ' --bvecs=' + input_bvec + ' --bvals=' + input_bval + ' --out='  + eddy_output_basename + ' --cnr_maps --residuals'
    else:
        command = 'OMP_NUM_THREADS='+str(nthreads)+ ' ' + eddy + ' --imain=' + input_dwi + ' --mask=' + tmp_mask + ' --index=' + input_index + ' --acqp=' + input_acqparam + ' --bvecs=' + input_bvec + ' --bvals=' + input_bval + ' --out='  + eddy_output_basename + ' --cnr_maps --residuals'

    command+= ' --ol_type='+ol_type

    if topup_base != '':
        command += ' --topup='+topup_base
    if external_b0 != '':
        command += ' --field='+external_b0
    if repol != 0:
        command += ' --repol '
    if data_shelled != 0:
        command += ' --data_is_shelled '
    if mb != '':
        command += ' --mb=' + mb + ' --mb_offs='+mb_off
    if slice_order != '':
        command += ' --slspec='+slice_order
        if mporder != 0:
            command += ' --niter=10 --fwhm=10,8,6,4,4,2,2,0,0,0 --mporder='+str(mporder)+' --s2v_niter=12 --s2v_lambda=6 --s2v_interp=spline'
            
    if estimate_move_by_suscept == True:
        command += ' --estimate_move_by_susceptibility'

    #print(command)
    os.system(command)
    #Rotate b-vecs after doing the eddy correction
    os.system('mv ' + eddy_output_basename+'.eddy_rotated_bvecs ' + output_bvec)

    #Remove temporary mask
    os.system('rm -rf ' + tmp_mask)
    if mask_img == '':
        os.system('rm -rf ' + tmp_dwi)

def compute_average_motion(eddy_basename):
    movement_rms_file = eddy_basename + '.eddy_movement_rms'
    restricted_movement_rms_file = eddy_basename + '.eddy_restricted_movement_rms'

    movement_rms = np.loadtxt(movement_rms_file)
    restricted_movement_rms = np.loadtxt(restricted_movement_rms_file)


    avg_movement_rms = np.mean(movement_rms, axis=0)
    avg_restricted_movement_rms = np.mean(restricted_movement_rms, axis=0)

    avg_global_displacement = avg_movement_rms[0]
    avg_slice_displacement  = avg_movement_rms[1]

    avg_restricted_displacement = avg_restricted_movement_rms[0]
    avg_restricted_slice_displacement = avg_restricted_movement_rms[1]

    return ('Average Total Movement', avg_movement_rms[0],
            'Average Slice Movement', avg_movement_rms[1],
            'Average Restricted Movement', avg_restricted_movement_rms[0],
            'Average Restricted Slice Movement', avg_restricted_movement_rms[1])

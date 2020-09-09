import string, os, sys, subprocess, shutil, time

import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti

from dipy.denoise.noise_estimate import estimate_sigma
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs, reorient_vectors
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.dti import fractional_anisotropy
from dipy.io.utils import nifti1_symmat

def calculate_dti_skewness(input_tensor, output_dir):
    tmp_dir = output_dir + '/tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    tmp_tensor = tmp_dir+'tensor.nii.gz'
    md = tmp_dir+'md.nii.gz'
    shutil.copy2(input_tensor, tmp_tensor)
    os.chdir(tmp_dir)
    os.system('TVEigenSystem -in ' + tmp_tensor + ' -type FSL')
    os.system('TVtool -in ' + tmp_tensor + ' -out ' + md + ' -tr')
    os.system('fslmaths ' + md + ' -div 3.00 ' + md)

    l1 = tmp_dir+'l1.nii.gz'
    l2 = tmp_dir+'l2.nii.gz'
    l3 = tmp_dir+'l3.nii.gz'

    l13 = tmp_dir+'l13.nii.gz'
    l23 = tmp_dir+'l23.nii.gz'
    l33 = tmp_dir+'l33.nii.gz'
    skewness = output_dir + 'dti_SKEWNESS.nii.gz'

    os.system('fslmaths ' + tmp_dir+'tensor_L1.nii.gz -sub ' + md + ' ' + l1)
    os.system('fslmaths ' + tmp_dir+'tensor_L2.nii.gz -sub ' + md + ' ' + l2)
    os.system('fslmaths ' + tmp_dir+'tensor_L3.nii.gz -sub ' + md + ' ' + l3)
    os.system('fslmaths ' + l1 + ' -mul ' + l1 + ' -mul ' + l1 + ' ' + l13)
    os.system('fslmaths ' + l2 + ' -mul ' + l2 + ' -mul ' + l2 + ' ' + l23)
    os.system('fslmaths ' + l3 + ' -mul ' + l3 + ' -mul ' + l3 + ' ' + l33)

    os.system('fslmaths ' + l13 + ' -add ' + l23 + ' -add ' + l33 + ' -div 3.0 ' + skewness)
    os.system('rm -rf ' + tmp_dir)

def fit_dti_dipy(input_dwi, input_bval, input_bvec, output_dir, fit_type='', mask='', bmax='', mask_tensor='F', bids_fmt=False, bids_id=''):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img = nib.load(input_dwi)
    axis_orient = nib.aff2axcodes(img.affine)
    
    ras_img = nib.as_closest_canonical(img)
    data = ras_img.get_data()
    
    bvals, bvecs = read_bvals_bvecs(input_bval, input_bvec)
    bvecs = reorient_vectors(bvecs, axis_orient[0]+axis_orient[1]+axis_orient[2] , 'RAS', axis=1)

    if mask != '':
        mask_img = nib.as_closest_canonical(nib.load(mask))
        mask_data = mask_img.get_data()

    if bmax != "":
        jj = np.where(bvals >= bmax)
        bvals = np.delete(bvals, jj)
        bvecs = np.delete(bvecs, jj, 0)
        data = np.delete(data, jj , axis=3)

    values = np.array(bvals)
    ii = np.where(values == bvals.min())[0]
    b0_average = np.mean(data[:,:,:,ii], axis=3)

    gtab = gradient_table(bvals, bvecs)

    if fit_type == 'RESTORE':
        sigma = estimate_sigma(data)
        #calculate the average sigma from the b0's
        sigma = np.mean(sigma[ii])

        dti_model = dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma)
        
        if mask != '':
            dti_fit = dti_model.fit(data, mask_data)
        else:
            dti_fit = dti_model.fit(data)

    elif fit_type != 'RESTORE' and fit_type != '':
        dti_model = dti.TensorModel(gtab, fit_method=fit_type)
        
        if mask != '':
            dti_fit = dti_model.fit(data, mask_data)
        else:
            dti_fit = dti_model.fit(data)

    else:
        dti_model = dti.TensorModel(gtab)
        
        if mask != '':
            dti_fit = dti_model.fit(data, mask_data)
        else:
            dti_fit = dti_model.fit(data)

    estimate_data = dti_fit.predict(gtab, S0=b0_average)
    residuals = np.absolute(data - estimate_data)

    tensor = dti.lower_triangular(dti_fit.quadratic_form.astype(np.float32))
    evecs = dti_fit.evecs.astype(np.float32)
    evals = dti_fit.evals.astype(np.float32)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_imgs = []

    #Define output imgs
    if bids_fmt:
        output_tensor_nifti     = output_dir + '/' + bids_id + '_model-DTI_parameter-TENSOR.nii.gz'
        output_tensor_fsl       = output_dir + '/' + bids_id + '_model-DTI_parameter-FSL_TENSOR.nii.gz'
        output_tensor_mrtrix    = output_dir + '/' + bids_id + '_model-DTI_parameter-MRTRIX_TENSOR.nii.gz'
    
        output_V1               = output_dir + '/' + bids_id + '_model-DTI_parameter-V1.nii.gz'
        output_V2               = output_dir + '/' + bids_id + '_model-DTI_parameter-V2.nii.gz'
        output_V3               = output_dir + '/' + bids_id + '_model-DTI_parameter-V3.nii.gz'
        output_FSL_V1           = output_dir + '/' + bids_id + '_model-DTI_parameter-FSL_V1.nii.gz'
        output_FSL_V2           = output_dir + '/' + bids_id + '_model-DTI_parameter-FSL_V2.nii.gz'
        output_FSL_V3           = output_dir + '/' + bids_id + '_model-DTI_parameter-FSL_V3.nii.gz'
        
        output_L1               = output_dir + '/' + bids_id + '_model-DTI_parameter-L1.nii.gz'
        output_L2               = output_dir + '/' + bids_id + '_model-DTI_parameter-L2.nii.gz'
        output_L3               = output_dir + '/' + bids_id + '_model-DTI_parameter-L3.nii.gz'
        
        output_fa               = output_dir + '/' + bids_id + '_model-DTI_parameter-FA.nii.gz'
        output_md               = output_dir + '/' + bids_id + '_model-DTI_parameter-MD.nii.gz'
        output_rd               = output_dir + '/' + bids_id + '_model-DTI_parameter-RD.nii.gz'
        output_ad               = output_dir + '/' + bids_id + '_model-DTI_parameter-AD.nii.gz'
        output_tr               = output_dir + '/' + bids_id + '_model-DTI_parameter-TRACE.nii.gz'
        
        output_ga               = output_dir + '/' + bids_id + '_model-DTI_parameter-GA.nii.gz'
        output_color_fa         = output_dir + '/' + bids_id + '_model-DTI_parameter-COLOR_FA.nii.gz'
        
        output_PL               = output_dir + '/' + bids_id + '_model-DTI_parameter-PLANARITY.nii.gz'
        output_SP               = output_dir + '/' + bids_id + '_model-DTI_parameter-SPHERICITY.nii.gz'
        output_MO               = output_dir + '/' + bids_id + '_model-DTI_parameter-MODE.nii.gz'
        
        output_res              = output_dir + '/' + bids_id + '_model-DTI_parameter-RESIDUALS.nii.gz'
        
    else:
        output_tensor_fsl           = output_dir + '/dti_FSL_TENSOR.nii.gz'
        output_tensor_nifti         = output_dir + '/dti_TENSOR.nii.gz'
        output_tensor_mrtrix        = output_dir + '/dti_MRTRIX_TENSOR.nii.gz'
        
        output_V1                   = output_dir + '/dti_V1.nii.gz'
        output_V2                   = output_dir + '/dti_V2.nii.gz'
        output_V3                   = output_dir + '/dti_V3.nii.gz'
        output_FSL_V1               = output_dir + '/dti_FSL_V1.nii.gz'
        output_FSL_V2               = output_dir + '/dti_FSL_V2.nii.gz'
        output_FSL_V3               = output_dir + '/dti_FSL_V3.nii.gz'
        
        output_L1                   = output_dir + '/dti_L1.nii.gz'
        output_L2                   = output_dir + '/dti_L2.nii.gz'
        output_L3                   = output_dir + '/dti_L3.nii.gz'

        output_fa               = output_dir + '/dti_FA.nii.gz'
        output_md               = output_dir + '/dti_MD.nii.gz'
        output_rd               = output_dir + '/dti_RD.nii.gz'
        output_ad               = output_dir + '/dti_AD.nii.gz'
        output_tr               = output_dir + '/dti_TRACE.nii.gz'
        
        output_ga               = output_dir + '/dti_GA.nii.gz'
        output_color_fa         = output_dir + '/dti_COLOR_FA.nii.gz'
        
        output_PL               = output_dir + '/dti_PLANARITY.nii.gz'
        output_SP               = output_dir + '/dti_SPHERICITY.nii.gz'
        output_MO               = output_dir + '/dti_MODE.nii.gz'
        
        output_res              = output_dir + '/dti_RESIDUALS.nii.gz'
        
        
    tensor_img = nifti1_symmat(tensor, ras_img.affine, ras_img.header)
    tensor_img.header.set_intent = 'NIFTI_INTENT_SYMMATRIX'
    tensor_img.to_filename(output_tensor_nifti)
    
    tensor_fsl          = np.empty(tensor.shape)
    tensor_fsl[:,:,:,0] = tensor[:,:,:,0]
    tensor_fsl[:,:,:,1] = tensor[:,:,:,1]
    tensor_fsl[:,:,:,2] = tensor[:,:,:,3]
    tensor_fsl[:,:,:,3] = tensor[:,:,:,2]
    tensor_fsl[:,:,:,4] = tensor[:,:,:,4]
    tensor_fsl[:,:,:,5] = tensor[:,:,:,5]
    save_nifti(output_tensor_fsl, tensor_fsl, ras_img.affine, ras_img.header)
    
    tensor_mrtrix           = np.empty(tensor.shape)
    tensor_mrtrix[:,:,:,0]  = tensor[:,:,:,0]
    tensor_mrtrix[:,:,:,1]  = tensor[:,:,:,2]
    tensor_mrtrix[:,:,:,2]  = tensor[:,:,:,5]
    tensor_mrtrix[:,:,:,3]  = tensor[:,:,:,1]
    tensor_mrtrix[:,:,:,4]  = tensor[:,:,:,3]
    tensor_mrtrix[:,:,:,5]  = tensor[:,:,:,4]
    save_nifti(output_tensor_mrtrix, tensor_mrtrix, ras_img.affine, ras_img.header)

    fa              = dti_fit.fa
    color_fa        = dti_fit.color_fa
    md              = dti_fit.md
    rd              = dti_fit.rd
    ad              = dti_fit.ad
    ga              = dti_fit.ga
    trace           = dti_fit.trace
    dti_mode        = dti_fit.mode
    dti_planarity   = dti_fit.planarity
    dti_sphericity  = dti_fit.sphericity

    #Remove any nan
    fa[np.isnan(fa)]                            = 0
    color_fa[np.isnan(color_fa)]                = 0
    md[np.isnan(md)]                            = 0
    rd[np.isnan(rd)]                            = 0
    ad[np.isnan(ad)]                            = 0
    ga[np.isnan(ga)]                            = 0
    trace[np.isnan(trace)]                      = 0
    dti_mode[np.isnan(dti_mode)]                = 0
    dti_planarity[np.isnan(dti_planarity)]      = 0
    dti_sphericity[np.isnan(dti_sphericity)]    = 0
    
    save_nifti(output_V1, evecs[:,:,:,:,0], ras_img.affine, ras_img.header)
    save_nifti(output_V2, evecs[:,:,:,:,1], ras_img.affine, ras_img.header)
    save_nifti(output_V3, evecs[:,:,:,:,2], ras_img.affine, ras_img.header)
    
    save_nifti(output_L1, evals[:,:,:,0], ras_img.affine, ras_img.header)
    save_nifti(output_L2, evals[:,:,:,1], ras_img.affine, ras_img.header)
    save_nifti(output_L3, evals[:,:,:,2], ras_img.affine, ras_img.header)

    save_nifti(output_fa, fa, ras_img.affine, ras_img.header)
    save_nifti(output_color_fa, color_fa, ras_img.affine, ras_img.header)
    save_nifti(output_md, md, ras_img.affine, ras_img.header)
    save_nifti(output_ad, ad, ras_img.affine, ras_img.header)
    save_nifti(output_rd, rd, ras_img.affine, ras_img.header)
    save_nifti(output_ga, ga, ras_img.affine, ras_img.header)
    save_nifti(output_tr, trace, ras_img.affine, ras_img.header)
    save_nifti(output_PL, dti_planarity, ras_img.affine, ras_img.header)
    save_nifti(output_SP, dti_sphericity, ras_img.affine, ras_img.header)
    save_nifti(output_MO, dti_mode, ras_img.affine, ras_img.header)
    save_nifti(output_res, residuals, ras_img.affine, ras_img.header)
    
    
    #Reorient back to the original
    output_imgs.append(output_tensor_nifti)
    output_imgs.append(output_tensor_fsl)
    output_imgs.append(output_tensor_mrtrix)
    output_imgs.append(output_V1)
    output_imgs.append(output_V2)
    output_imgs.append(output_V3)
    output_imgs.append(output_L1)
    output_imgs.append(output_L2)
    output_imgs.append(output_L3)
    output_imgs.append(output_fa)
    output_imgs.append(output_md)
    output_imgs.append(output_rd)
    output_imgs.append(output_ad)
    output_imgs.append(output_ga)
    output_imgs.append(output_color_fa)
    output_imgs.append(output_PL)
    output_imgs.append(output_SP)
    output_imgs.append(output_MO)
    output_imgs.append(output_res)
    
    #Change orientation back to the original orientation
    orig_ornt   = nib.io_orientation(ras_img.affine)
    targ_ornt   = nib.io_orientation(img.affine)
    transform   = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    affine_xfm  = nib.orientations.inv_ornt_aff(transform, ras_img.shape)
    trans_mat = affine_xfm[0:3,0:3]
    
    for img_path in output_imgs:
        orig_img    = nib.load(img_path)
        reoriented  = orig_img.as_reoriented(transform)
        reoriented.to_filename(img_path)
        
    #Correct FSL tensor for orientation
    dirs = []
    dirs.append(np.array([[1],[0],[0]]))
    dirs.append(np.array([[1],[1],[0]]))
    dirs.append(np.array([[1],[0],[1]]))
    dirs.append(np.array([[0],[1],[0]]))
    dirs.append(np.array([[0],[1],[1]]))
    dirs.append(np.array([[0],[0],[1]]))
    
    tensor_fsl = nib.load(output_tensor_fsl)
    corr_fsl_tensor = np.empty(tensor_fsl.get_data().shape)
    
    for i in range(0,len(dirs)):
        
        rot_dir = np.matmul(trans_mat, dirs[i])
        sign = 1.0
        if np.sum(rot_dir) == 0.0:
            sign = -1.0
        
        if (np.absolute(rot_dir) == np.array([[1],[0],[0]])).all():
            tensor_ind = 0
        elif (np.absolute(rot_dir) == np.array([[1],[1],[0]])).all():
            tensor_ind = 1
        elif (np.absolute(rot_dir) == np.array([[1],[0],[1]])).all():
            tensor_ind = 2
        elif (np.absolute(rot_dir) == np.array([[0],[1],[0]])).all():
            tensor_ind = 3
        elif ( np.absolute(rot_dir) == np.array([[0],[1],[1]])).all():
            tensor_ind = 4
        elif ( np.absolute(rot_dir) == np.array([[0],[0],[1]])).all():
            tensor_ind = 5
        
        corr_fsl_tensor[:,:,:,i] = sign*tensor_fsl.get_data()[:,:,:,tensor_ind]
    
    save_nifti(output_tensor_fsl, corr_fsl_tensor, tensor_fsl.affine, tensor_fsl.header)
    
    #Now correct the eigenvectors
    #Determine the order to rearrange
    vec_order = np.transpose(targ_ornt[:,0]).astype(int)
    sign_order = np.transpose(targ_ornt[:,1]).astype(int)
    
    fsl_v1 = nib.load(output_V1)
    corr_fsl_v1 = fsl_v1.get_data()[:,:,:,vec_order]
    for i in range(0,2):
        corr_fsl_v1[:,:,:,i] = sign_order[i]*corr_fsl_v1[:,:,:,i]
        
    save_nifti(output_FSL_V1, corr_fsl_v1, fsl_v1.affine, fsl_v1.header)
    
    
    fsl_v2 = nib.load(output_V2)
    corr_fsl_v2 = fsl_v2.get_data()[:,:,:,vec_order]
    for i in range(0,2):
        corr_fsl_v2[:,:,:,i] = sign_order[i]*corr_fsl_v2[:,:,:,i]
        
    save_nifti(output_FSL_V2, corr_fsl_v2, fsl_v2.affine, fsl_v2.header)
    
    
    fsl_v3 = nib.load(output_V3)
    corr_fsl_v3 = fsl_v3.get_data()[:,:,:,vec_order]
    for i in range(0,2):
        corr_fsl_v3[:,:,:,i] = sign_order[i]*corr_fsl_v3[:,:,:,i]
         
    save_nifti(output_FSL_V3, corr_fsl_v3, fsl_v3.affine, fsl_v3.header)
        
        
    
def fit_dti_mrtrix(input_dwi, input_bval, input_bvec, output_dir, mask='', bmax=''):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_tensor = output_dir + '/dti_tensor.nii.gz'
    output_V1 = output_dir + '/dti_V1.nii.gz'
    output_V2 = output_dir + '/dti_V2.nii.gz'
    output_V3 = output_dir + '/dti_V3.nii.gz'
    output_L1 = output_dir + '/dti_L1.nii.gz'
    output_L2 = output_dir + '/dti_L2.nii.gz'
    output_L3 = output_dir + '/dti_L3.nii.gz'

    output_fa = output_dir + '/dti_FA.nii.gz'
    output_md = output_dir + '/dti_MD.nii.gz'
    output_rd = output_dir + '/dti_RD.nii.gz'
    output_ad = output_dir + '/dti_AD.nii.gz'
    
    if bmax!='':
        img = nib.load(input_dwi)
        data = img.get_data()
        bvals, bvecs = read_bvals_bvecs(input_bval, input_bvec)

        aff = img.get_affine()
        sform = img.get_sform()
        qform = img.get_qform()
    
        jj = np.where(bvals >= bmax)
        bvals = np.delete(bvals, jj)
        bvecs = np.delete(bvecs, jj, 0)
        data = np.delete(data, jj , axis=3)

        #Save the dwi data
        tmp_dwi_img = nib.Nifti1Image(data,aff,img.header)
        tmp_dwi_img.set_sform(sform)
        tmp_dwi_img.set_qform(qform)
        nib.save(tmp_dwi_img, output_dir+'/tmp_dwi.nii.gz')
        np.savetxt(output_dir+'/tmp_bvals.bval', bvals, fmt='%i')
        np.savetxt(output_dir+'/tmp_bvecs.bvec', np.transpose(bvecs), fmt='%.5f')

        #Run the tensor fitting using MRTRIX:
        command = 'dwi2tensor -fslgrad ' + output_dir+'/tmp_bvecs.bvec ' + output_dir+'/tmp_bvals.bval ' + output_dir+'/tmp_dwi.nii.gz ' + output_tensor

    else:
        command = 'dwi2tensor -fslgrad ' + input_bvec + ' ' +  input_bval + ' ' + input_dwi + ' ' + output_tensor

    if mask!='':
        os.system(command+' -mask ' + mask)
    else:
        os.system(command)

    #Write out the parameters
    os.system('tensor2metric -adc ' + output_md + ' ' + output_tensor)
    os.system('tensor2metric -fa ' + output_fa + ' ' + output_tensor)
    os.system('tensor2metric -ad ' + output_ad + ' ' + output_tensor)
    os.system('tensor2metric -rd ' + output_rd + ' ' + output_tensor)


    os.system('tensor2metric -value ' + output_L1 + ' -num 1 ' + output_tensor)
    os.system('tensor2metric -value ' + output_L2 + ' -num 2 ' + output_tensor)
    os.system('tensor2metric -value ' + output_L3 + ' -num 3 ' + output_tensor)

    os.system('tensor2metric -vector ' + output_V1 + ' -num 1 ' + output_tensor)
    os.system('tensor2metric -vector ' + output_V2 + ' -num 2 ' + output_tensor)
    os.system('tensor2metric -vector ' + output_V3 + ' -num 3 ' + output_tensor)

    os.system('rm -rf ' + output_dir + '/tmp*')

#def fit_dti_camino(input_dwi, input_bval, input_bvec, output_dir, fit_type='', mask='', bmax=''):
#    
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#
#    #First create temporary camino style data
#    camino_dwi = output_dir + '/tmp_dwi.Bfloat'
#    camino_scheme = output_dir + '/tmp_dwi.scheme'
#    camino_tensor = output_dir + '/tmp_dti.Bfloat'
#    os.system('image2voxel -4dimage ' + input_dwi + ' -outputfile ' + camino_dwi)
#    os.system('fsl2scheme -bvecfile ' + input_bvec + ' -bvalfile ' + input_bval + ' > ' + camino_scheme)
#
#    if fit_type == 'RESTORE':
#        data = nib.load(input_dwi).get_data()
#        bvals, bvecs = read_bvals_bvecs(input_bval, input_bvec)
#        values = np.array(bvals)
#        ii = np.where(values == bvals.min())[0]
#        sigma = estimate_sigma(data)
#        sigma = np.mean(sigma[ii])
#
#        #FIT TENSOR
#        os.system('modelfit -inputfile ' + camino_dwi + ' -schemefile ' + camino_scheme + ' -model restore -sigma ' + str(sigma) + ' -bgmask ' + mask + ' -outputfile ' + camino_tensor)
#
#    elif fit_type == 'WLLS':
#        os.system('modelfit -inputfile ' + camino_dwi + ' -schemefile ' + camino_scheme + ' -model ldt_wtd -bgmask ' + mask + ' -outputfile ' + camino_tensor)
#                  
#    elif fit_type == 'NLLS':
#        os.system('modelfit -inputfile ' + camino_dwi + ' -schemefile ' + camino_scheme + ' -model nldt_pos -bgmask ' + mask + ' -outputfile ' + camino_tensor)
#                  
#    else:
#        os.system('modelfit -inputfile ' + camino_dwi + ' -schemefile ' + camino_scheme + ' -model ldt -bgmask ' + mask + ' -outputfile ' + camino_tensor)
#                
#    #Convert the data back to NIFTI
#    output_root = output_dir + 'dti_'
#    os.system('dt2nii -inputfile ' + camino_tensor + ' -gzip -inputdatatype double -header ' + input_dwi + ' -outputroot ' + output_root)
#
#    #Define the output file paths
#    output_tensor = output_dir + '/dti_tensor.nii.gz'
#    output_tensor_spd = output_dir + '/dti_tensor_spd.nii.gz'
#    output_tensor_norm = output_dir + '/dti_tensor_norm.nii.gz'
#    norm_mask = output_dir + '/norm_mask.nii.gz'
#    output_tensor_spd_masked = output_dir + '/dti_tensor_spd_masked.nii.gz'
#
#    output_V1 = output_dir + '/dti_V1.nii.gz'
#    output_V2 = output_dir + '/dti_V2.nii.gz'
#    output_V3 = output_dir + '/dti_V3.nii.gz'
#    output_L1 = output_dir + '/dti_L1.nii.gz'
#    output_L2 = output_dir + '/dti_L2.nii.gz'
#    output_L3 = output_dir + '/dti_L3.nii.gz'
#    
#    output_fa = output_dir + '/dti_FA.nii.gz'
#    output_md = output_dir + '/dti_MD.nii.gz'
#    output_rd = output_dir + '/dti_RD.nii.gz'
#    output_ad = output_dir + '/dti_AD.nii.gz'
#    
#    output_res = output_dir + '/dti_residuals.nii.gz'
#
#    os.system('TVtool -in ' + output_root + 'dt.nii.gz -scale 1e9 -out ' + output_tensor)
#    os.system('TVtool -in ' + output_tensor + ' -spd -out ' + output_tensor_spd)
#    os.system('TVtool -in ' + output_tensor_spd + ' -norm -out ' + output_tensor_norm)
#    os.system('BinaryThresholdImageFilter ' +  output_tensor_norm + ' ' + norm_mask + ' 0.01 3.0 1 0')
#    os.system('TVtool -in ' + output_tensor_spd + ' -mask ' + norm_mask + ' -out ' + output_tensor_spd_masked)
#    os.system('TVFromEigenSystem -basename dti -type FSL -out ' + output_tensor_spd_masked)
#
#    #Calculate FA, MD, RD, AD
#    os.system('TVtool -in ' + output_tensor_spd_masked + ' -fa -out ' + output_fa)
#    os.system('TVtool -in ' + output_tensor_spd_masked + ' -rd -out ' + output_rd)
#    os.system('TVtool -in ' + output_tensor_spd_masked + ' -ad -out ' + output_ad)
#    os.system('TVtool -in ' + output_tensor_spd_masked + ' -tr -out ' + output_md)
#    os.system('fslmaths ' + output_md + ' -div 3.00 ' + output_md)
#
#    #Output the eigenvectors and eigenvalues
#    os.system('TVEigenSystem -in ' + output_tensor_spd_masked + ' -type FSL')
#    dti_basename=nib.filename_parser.splitext_addext(output_tensor_spd_masked)[0]
#    os.system('mv ' + dti_basename + '_V1.nii.gz ' + output_V1)
#    os.system('mv ' + dti_basename + '_V2.nii.gz ' + output_V2)
#    os.system('mv ' + dti_basename + '_V3.nii.gz ' + output_V3)
#    os.system('mv ' + dti_basename + '_L1.nii.gz ' + output_L1)
#    os.system('mv ' + dti_basename + '_L2.nii.gz ' + output_L2)
#    os.system('mv ' + dti_basename + '_L3.nii.gz ' + output_L3)
#
#    #Clean up files
#    os.system('rm -rf ' + dti_basename +'_[V,L]* ' + output_dir + '/tmp*')


def fit_fwdti_dipy(input_dwi, input_bval, input_bvec, output_dir, fit_method='', mask=''):

    import dipy.reconst.fwdti as fwdti
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if fit_method=='':
        fit_method = 'WLS'

    img = nib.load(input_dwi)
    data = img.get_data()
    bvals, bvecs = read_bvals_bvecs(input_bval, input_bvec,)
    gtab = gradient_table(bvals, bvecs)

    if mask != '':
        mask_data = nib.load(mask).get_data()

    values = np.array(bvals)
    ii = np.where(values == bvals.min())[0]
    b0_average = np.mean(data[:,:,:,ii], axis=3)
    
    fwidtimodel = fwdti.FreeWaterTensorModel(gtab, fit_method)

    if mask!='':
        fwidti_fit = fwidtimodel.fit(data, mask_data)
    else:
        fwidti_fit = fwidtimodel.fit(data)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_evecs = output_dir + '/fwe_dti_eigenvectors.nii.gz'
    output_evals = output_dir + '/fwe_dti_eigenvalues.nii.gz'

    output_fa = output_dir + '/fwe_dti_FA.nii.gz'
    output_md = output_dir + '/fwe_dti_MD.nii.gz'
    output_rd = output_dir + '/fwe_dti_RD.nii.gz'
    output_ad = output_dir + '/fwe_dti_AD.nii.gz'
    output_f = output_dir + '/fwe_dti_F.nii.gz'

    #Calculate Parameters for FWDTI Model
    evals_img = nib.Nifti1Image(fwidti_fit.evals.astype(np.float32), img.get_affine(),img.header)
    nib.save(evals_img, output_evals)
    
    evecs_img = nib.Nifti1Image(fwidti_fit.evecs.astype(np.float32), img.get_affine(),img.header)
    nib.save(evecs_img, output_evecs)
    
    fwidti_fa = fwidti_fit.fa
    fwidti_fa_img = nib.Nifti1Image(fwidti_fa.astype(np.float32), img.get_affine(),img.header)
    nib.save(fwidti_fa_img, output_fa)
    
    fwidti_md = fwidti_fit.md
    fwidti_md_img = nib.Nifti1Image(fwidti_md.astype(np.float32), img.get_affine(),img.header)
    nib.save(fwidti_md_img, output_md)

    fwidti_ad = fwidti_fit.ad
    fwidti_ad_img = nib.Nifti1Image(fwidti_ad.astype(np.float32), img.get_affine(),img.header)
    nib.save(fwidti_ad_img, output_ad)
    
    fwidti_rd = fwidti_fit.rd
    fwidti_rd_img = nib.Nifti1Image(fwidti_rd.astype(np.float32), img.get_affine(),img.header)
    nib.save(fwidti_rd_img, output_rd)

    fwidti_f = fwidti_fit.f
    fwidti_f_img = nib.Nifti1Image(fwidti_f.astype(np.float32), img.get_affine(),img.header)
    nib.save(fwidti_f_img, output_f)


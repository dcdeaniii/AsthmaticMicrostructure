import string, os, sys, subprocess, shutil, time
import nibabel as nib
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed, SD2BinghamDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.core import modeling_framework
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dipy.io import read_bvals_bvecs


model_type = 'WATSON'
parallel_diffusivity = 1.7e-9
iso_diffusivity = 3e-9
nthreads = 1
solver = 'brute2fine'

input = sys.argv[1:]
input_dwi = input[0]
input_bval = input[1]
input_bvec = input[2]
input_mask = input[3]
ouput_dir = input[4]


if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#Setup the acquisition scheme
bvals, bvecs = read_bvals_bvecs(input_bval, input_bvec)
bvals_SI = bvals*1e6
acq_scheme = acquisition_scheme_from_bvalues(bvals_SI, bvecs)
acq_scheme.print_acquisition_info

#Load the data
img = nib.load(input_dwi)
data = img.get_data()

#Load the mask
img = nib.load(input_mask)
mask_data = img.get_data()

ball = gaussian_models.G1Ball() #CSF
stick = cylinder_models.C1Stick() #Intra-axonal diffusion
zeppelin = gaussian_models.G2Zeppelin() #Extra-axonal diffusion

if model_type == 'Bingham' or model_type == 'BINGHAM':
    dispersed_bundle = SD2BinghamDistributed(models=[stick, zeppelin])
else:
    dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])

dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', parallel_diffusivity)

NODDI_mod = MultiCompartmentModel(models=[ball, dispersed_bundle])
NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', iso_diffusivity)
NODDI_fit = NODDI_mod.fit(acq_scheme, data, mask=mask_data, number_of_processors=nthreads, solver=solver)

fitted_parameters = NODDI_fit.fitted_parameters

if model_type == 'Bingham' or model_type == 'BINGHAM':
    # get total Stick signal contribution
    vf_intra = (fitted_parameters['SD2BinghamDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])

    # get total Zeppelin signal contribution
    vf_extra = ((1 - fitted_parameters['SD2BinghamDistributed_1_partial_volume_0'])*fitted_parameters['partial_volume_1'])
    vf_iso = fitted_parameters['partial_volume_0']
    odi = fitted_parameters['SD2BinghamDistributed_1_SD2Bingham_1_odi']

else:
    # get total Stick signal contribution
    vf_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])

    # get total Zeppelin signal contribution
    vf_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'])*fitted_parameters['partial_volume_1'])
    vf_iso = fitted_parameters['partial_volume_0']
    odi = fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']

if bids_fmt:
    output_odi      = output_dir + '/' + bids_id + '_model-NODDI_parameter-ODI.nii.gz'
    output_vf_intra = output_dir + '/' + bids_id + '_model-NODDI_parameter-ICVF.nii.gz'
    output_vf_extra = output_dir + '/' + bids_id + '_model-NODDI_parameter-EXVF.nii.gz'
    output_vf_iso   = output_dir + '/' + bids_id + '_model-NODDI_parameter-ISO.nii.gz'
else:
    output_odi      = output_dir + '/noddi_ODI.nii.gz'
    output_vf_intra = output_dir + '/noddi_ICVF.nii.gz'
    output_vf_extra = output_dir + '/noddi_EXVF.nii.gz'
    output_vf_iso   = output_dir + '/noddi_ISO.nii.gz'

#Save the images
odi_img = nib.Nifti1Image(odi,img.get_affine(),img.header)
odi_img.set_sform(img.get_sform())
odi_img.set_qform(img.get_qform())
nib.save(odi_img, output_odi)

icvf_img = nib.Nifti1Image(vf_intra,img.get_affine(),img.header)
icvf_img.set_sform(img.get_sform())
icvf_img.set_qform(img.get_qform())
nib.save(icvf_img, output_vf_intra)

ecvf_img = nib.Nifti1Image(vf_extra,img.get_affine(),img.header)
ecvf_img.set_sform(img.get_sform())
ecvf_img.set_qform(img.get_qform())
nib.save(ecvf_img, output_vf_extra)

iso_img = nib.Nifti1Image(vf_iso, img.get_affine(), img.header)
iso_img.set_sform(img.get_sform())
iso_img.set_qform(img.get_qform())
nib.save(iso_img, output_vf_iso)

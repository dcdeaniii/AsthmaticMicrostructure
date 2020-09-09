import string, os, sys, subprocess, shutil, time
from glob import glob

import numpy as np
import nibabel as nib

def rigid(input_image, ref_image, output_image, output_transform):

    output_base = os.path.basename(os.path.abspath(output_transform))
    os.system('ANTS 3 -o ' + output_base + ' -m MI[' + ref_image+','+input_image+',1,32] -i 0 --use-Histogram-Matching --number-of-affine-iterations 1000x1000x1000x100x100 --rigid-affine true  --do-rigid --affine-gradient-descent-option  0.5x0.95x1.e-4x1.e-4')

    os.rename(output_base+'Affine.txt', output_transform)
    os.system('antsApplyTransforms -d 3 -i ' + input_image + ' -o ' + output_image + ' -r ' + ref_image + ' -t ' + output_transform)

def affine(input_image, ref_iamge, output_image, output_transform):
    output_base = os.path.basename(os.path.abspath(output_transform))
    os.system('ANTS 3 -o ' + output_base + ' -m MI[' + ref_image+','+input_image+',1,32] -i 0 --use-Histogram-Matching --number-of-affine-iterations 1000x1000x1000x100x100 --rigid-affine false  --affine-gradient-descent-option  0.5x0.95x1.e-4x1.e-4')
    
    os.rename(output_base+'Affine.txt', output_transform)
    os.system('antsApplyTransforms -d 3 -i ' + input_image + ' -o ' + output_image + ' -r ' + ref_image + ' -t ' + output_transform)


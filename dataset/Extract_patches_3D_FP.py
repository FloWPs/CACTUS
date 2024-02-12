"""
Goal :
------------
Extract false alarms vignettes along 3 orthogonal projections (Axial, Sagittal & Coronal)

Steps :
------------
1. Load volumes
2. Load predictions
2. Locate false alarms as 3D connected components
3. Extract 3 orthogonal slices around the 3D components
"""


#####################
# IMPORT LIBRAIRIES #
#####################


import os
import random
import pandas as pd
import numpy as np

from utils import get_centroids, zscore_percentile, extract_vignette


#####################
#   CONFIGURATION   #
#####################

# Training data (train + valid)
train_patlist = [i for i in range(60)]

DATA = np.load('./meta_patient_T1_FLAIR_SPM12.npy')[train_patlist, :, :, :, 1] # FLAIR images
GT = np.load('./meta_patient_ground_truth_T1_FLAIR_SPM12.npy')[train_patlist, :, :, :]

# Predictions from which to extract false positive patches
PATH_PRED = 'D:/transunet_pytorch/results/WMH'

# Parameters for patch extraction
VIGNETTE_DIM =(32, 32)
SAMPLING_FACTOR = 6

PATH_FALSE_ALARM = './false_alarms/'
if not os.path.isdir(PATH_FALSE_ALARM):
    os.makedirs(PATH_FALSE_ALARM)


######################################
#    FALSE ALARMS PATCH EXTRACTION   #
######################################


# Number of patches extracted
row_list = []
tot_vignettes = 0


for patient in range(len(DATA)):

    # Load volumes
    volume = DATA[patient, 12:140, 30:-31, 4:-4]
    mask = GT[patient, 12:140, 30:-31, 4:-4]

    # Load predictions for each projection
    TU_mask_A = np.load(os.path.join(PATH_PRED, f'Patient{patient}', 'pred_TU_3S_axial.npy'))
    TU_mask_S = np.load(os.path.join(PATH_PRED, f'Patient{patient}', 'pred_TU_3S_sagittal.npy'))
    TU_mask_C = np.load(os.path.join(PATH_PRED, f'Patient{patient}', 'pred_TU_3S_coronal.npy'))
    
    # Union of predictions (to maximize the number of
    # false positives encountered along the 3 projections)
    TU_mask = TU_mask_A | TU_mask_S | TU_mask_C

    PATH_VIGNETTES = os.path.join(PATH_FALSE_ALARM, f'Patient{patient}')
    if not os.path.isdir(PATH_VIGNETTES):
        os.makedirs(PATH_VIGNETTES)

    cc = 0

    # Scans volume slices along the axial dimension
    for a in range(TU_mask.shape[2]):
        image = volume[:, :, a]
        gt = mask[:, :, a]
        pred = TU_mask[:, :, a]

        if np.max(TU_mask) > 0:
            # Get centroids of false alarms
            centroids = get_centroids(pred, cc_area_min=0)

            for j in range(len(centroids)):
                s = int(centroids[j][1]) # Order is reversed
                c = int(centroids[j][0])
                # print(a, s, c)

                # Get the 3 corresponding A+S+C orthogonal slices
                element_image_S = volume[s, :, :]
                element_image_C = volume[:, c, :]
                element_image_A = volume[:, :, a]

                # Normalize values
                element_image_S = zscore_percentile(element_image_S)
                element_image_C = zscore_percentile(element_image_C)
                element_image_A = zscore_percentile(element_image_A)

                # Ensure that there are not too many adjacent slices in the database
                if random.randint(0, SAMPLING_FACTOR) == SAMPLING_FACTOR:

                    # Extract the 3 corresponding A+S+C orthogonal vignettes
                    # centered around the connected component
                    vignette_A = extract_vignette(element_image_A, [s, c], dimensions=VIGNETTE_DIM)
                    vignette_S = extract_vignette(element_image_S, [c, a], dimensions=VIGNETTE_DIM)
                    vignette_C = extract_vignette(element_image_C, [s, a], dimensions=VIGNETTE_DIM)

                    # Extract the same ROI from the groundtruth mask
                    mask_A = extract_vignette(mask[:, :, a], [s, c], dimensions=VIGNETTE_DIM)
                    mask_S = extract_vignette(mask[s, :, :], [c, a], dimensions=VIGNETTE_DIM)
                    mask_C = extract_vignette(mask[:, c, :], [s, a], dimensions=VIGNETTE_DIM)

                    # If the connected component is a false positive
                    if np.max(mask_A)<1 and np.max(mask_S)<1 and np.max(mask_C)<1:

                        # Concatenate the results into a single image with 3 channels
                        vignette_3D = np.zeros((VIGNETTE_DIM[0], VIGNETTE_DIM[1], 3), dtype=np.float32)
                        vignette_3D[:, :, 0] = vignette_A
                        vignette_3D[:, :, 1] = vignette_S
                        vignette_3D[:, :, 2] = vignette_C
                        
                        # Save 3D patch
                        np.save(os.path.join(PATH_VIGNETTES, 'FA_3D_'+str(a)+'_cc-'+str(j+1)+'.npy'), vignette_3D.astype(np.float32))
                        cc += 1

    # Monitoring of the number of vignettes generatad for each patient
    stats = {'PatientID': f'Patient{patient}', 'Nombre de vignettes 3D': cc}
    row_list.append(stats)
    tot_vignettes = tot_vignettes + cc

print(f'Total number of 3D vignettes generated : {tot_vignettes}')

df_stats = pd.DataFrame(row_list, columns=['PatientID', 'Nombre de vignettes 3D'])
df_stats.to_csv(os.path.join(PATH_FALSE_ALARM, 'FP_stats.csv'))
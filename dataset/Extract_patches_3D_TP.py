"""
Goal :
------------
Extract PWML vignettes along 3 orthogonal projections (Axial, Sagittal & Coronal)

Steps :
------------
1. Load volumes
2. Locate PWML as 3D connected components
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

# Parameters for patch extraction
VIGNETTE_DIM =(32, 32)
SAMPLING_FACTOR = 1

PATH_TRUE_ALARM = './true_alarms/'
if not os.path.isdir(PATH_TRUE_ALARM):
    os.makedirs(PATH_TRUE_ALARM)


#####################
#    DATA LOADING   #
#####################

VIGNETTE_DIM = (32, 32) # (16, 16) #

# Number of vignette generated
tot_vignettes = 0
row_list = []

for patient in range(len(DATA)):

    # Load volumes
    volume = DATA[patient, 12:140, 30:-31, 15:116]
    mask = GT[patient, 12:140, 30:-31, 15:116]

    PATH_VIGNETTES = os.path.join(PATH_TRUE_ALARM, f'Patient{patient}')
    if not os.path.isdir(PATH_VIGNETTES):
        os.makedirs(PATH_VIGNETTES)

    cc = 0

    # Scans volume slices along the axial dimension
    for a in range(volume.shape[2]): 
        element_mask = mask[:, :, a]

        if np.max(element_mask) > 0:
            # Get centroids of true alarms on the axial slice (S + C coordinates)
            centroids = get_centroids(element_mask, cc_area_min=0)

            for j in range(len(centroids)):
                s = int(centroids[j][1]) # Order is reversed
                c = int(centroids[j][0])
                
                # Get the 3 corresponding A+S+C orthogonal slices
                element_image_S = volume[s, :, :]
                element_image_C = volume[:, c, :]
                element_image_A = volume[:, :, a]

                # Normalize values
                element_image_S = zscore_percentile(element_image_S)
                element_image_C = zscore_percentile(element_image_C)
                element_image_A = zscore_percentile(element_image_A)

                # Extract the 3 corresponding A+S+C orthogonal vignettes
                vignette_A = extract_vignette(element_image_A, [s, c], dimensions=VIGNETTE_DIM)
                vignette_S = extract_vignette(element_image_S, [c, a], dimensions=VIGNETTE_DIM)
                vignette_C = extract_vignette(element_image_C, [s, a], dimensions=VIGNETTE_DIM)

                mask_A = extract_vignette(mask[:, :, a], [s, c], dimensions=VIGNETTE_DIM)
                mask_S = extract_vignette(mask[s, :, :], [c, a], dimensions=VIGNETTE_DIM)
                mask_C = extract_vignette(mask[:, c, :], [s, a], dimensions=VIGNETTE_DIM)

                # # Rotate images to be vertical
                # vignette_C = rotate(vignette_C, angle=90)
                # vignette_A = rotate(vignette_A, angle=90)

                # Concatenate the results into a single image with 3 channels
                vignette_3D = np.zeros((VIGNETTE_DIM[0], VIGNETTE_DIM[1], 3), dtype=np.float32)
                vignette_3D[:, :, 0] = vignette_A
                vignette_3D[:, :, 1] = vignette_S
                vignette_3D[:, :, 2] = vignette_C

                # Ensure that there are not too many adjacent slices in the database
                if random.randint(0, SAMPLING_FACTOR) == SAMPLING_FACTOR:
                    # Save image
                    np.save(os.path.join(PATH_VIGNETTES, 'TA_3D_'+str(a)+'_cc-'+str(j+1)+'.npy'), vignette_3D.astype(np.float32))
                    cc += 1
    

    # Monitoring of the number of vignettes generatad for each patient
    stats = {'PatientID': f'Patient{patient}', 'Nombre de vignettes 3D': cc}
    # print(stats)
    row_list.append(stats)
    tot_vignettes = tot_vignettes + cc

print(f'Total number of 3D vignettes generated : {tot_vignettes}')

df_stats = pd.DataFrame(row_list, columns=['PatientID', 'Nombre de vignettes 3D'])
df_stats.to_csv(os.path.join(PATH_TRUE_ALARM, 'TP_stats.csv'))
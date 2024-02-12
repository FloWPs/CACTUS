"""
Goal:
------

Reduce the number of false alarms in the mask predicted
by the segmentation model to improve precision.

Steps:
------

1. Import volumes (images + multimodal predictions from TransUNet)
2. Extract patch around the connected components of the TransUNet-predicted mask
3. Save patch coordinates within the mask (for later repositioning)
4. Predict patch class
5. Modify the mask predicted by TransUNet according to the classification output
6. Performance assessment with groundtruth.
"""

#####################
# IMPORT LIBRAIRIES #
#####################

import os
import numpy as np
import pandas as pd
import hdf5storage

import cc3d

import torch
from utils.cactus import CACTUS

from utils.utils import extract_vignette, get_scores
from utils.utils import remove_small_lesions_threshold, remove_large_lesions_threshold
from utils.utils import zscore_percentile, read_and_preprocess_multiview
from config import cfg


#####################
#   CONFIGURATION   #
#####################


# Predictions from which to extract false positive patches
PATH_PRED = 'D:/transunet_pytorch/results/WMH'

# Models path
MODEL_PATH = cfg.inference.model_path
projection = cfg.inference.projection

# Datapaths
DATA = np.load(cfg.volumes)[:, :, :, :, 1] # FLAIR
GT = np.load(cfg.gt)
BRAIN_SEG = np.load(cfg.brain_seg)

# List of patients
PATLIST = cfg.test_patlist

# Vignette dimensions
VIGNETTE_DIM = (cfg.cactus.image_size, cfg.cactus.image_size)

MIN_LESION_SIZE = 50 # voxels
MAX_LESION_SIZE = 10000


##################
#   EVALUATION   #
##################


row_list = []
lesion_list = []

stat0 = []
stat0log = []
stat1 = []
stat1log = []

activation = {}


print(f'[THRESHOLD] {cfg.inference.threshold}')

patlist = cfg.test_patlist

for patient in patlist:
    print(f'Patient{patient}')
    
    PATH_PAT = os.path.join('results', f'Patient{patient}', 'correction_pred')
    if not os.path.isdir(PATH_PAT):
            os.makedirs(PATH_PAT)

    #------------------
    #   Data Loading
    #------------------

    # Load volume
    volume = DATA[patient, 12:140, 30:-31, 4:-4]

    # Load grountruth & brain segmentation
    gt_mask = GT[patient, 12:140, 30:-31, 4:-4]
    brain = BRAIN_SEG[patient, 12:140, 30:-31, 4:-4]
    
    # Threshold lesions based on size
    gt_mask = remove_small_lesions_threshold(gt_mask, min_lesion_size=5)
    # gt_mask = remove_large_lesions_threshold(gt_mask, max_lesion_size=MAX_LESION_SIZE)

    # Load segmented mask
    pred_mask = np.load(os.path.join(PATH_PRED, f'Patient{patient}', f'pred_TU_3S_coronal_t-0.01.npy'))
    pred_mask = pred_mask*1

    # Apply brain mask
    pred_mask[np.where(brain == 0)] = 0

    # Remove connected components based on size
    # pred_mask = remove_small_lesions_threshold(pred_mask, min_lesion_size=50)#MIN_LESION_SIZE)#MIN_LESION_SIZE)
    # pred_mask = remove_large_lesions_threshold(gt_mask, max_lesion_size=MAX_LESION_SIZE)

    
    #--------------------
    #   Model Loading
    #--------------------

    model = CACTUS(
            image_size = cfg.cactus.image_size,
            channels = cfg.cactus.channels,
            num_classes = cfg.cactus.num_classes,
            patch_size_axial = cfg.cactus.patch_size_axial, 
            patch_size_coronal = cfg.cactus.patch_size_coronal,
            axial_dim = cfg.cactus.axial_dim,
            coronal_dim = cfg.cactus.coronal_dim, 
            axial_depth = cfg.cactus.axial_depth, 
            coronal_depth = cfg.cactus.coronal_depth,
            cross_attn_depth = cfg.cactus.cross_attn_depth,
            multi_scale_enc_depth = cfg.cactus.multi_scale_enc_depth,
            heads = cfg.cactus.heads
            )

    checkpoint = torch.load(MODEL_PATH)['model_state_dict']
    model.load_state_dict(checkpoint)
    
    #----------------------------------------------------
    #    Connected Component Extraction & Prediction
    #----------------------------------------------------

    pred_mask_correction = pred_mask.copy()

    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(pred_mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesions found.')
    else:
        print(N, 'lesions found.')

    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # First element is always the background
    del lesions[0]

    # List of centroids per lesion cluster with segid as key
    centroids = dict(enumerate(stats['centroids']))
    # First element is always the background
    del centroids[0]

    for k in range(1, len(lesions)+1):
        # print(lesions[k], centroids[k])
        if lesions[k] > 0:
            vignetteA = extract_vignette(zscore_percentile(volume[:, :, int(centroids[k][2])]), [int(centroids[k][0]), int(centroids[k][1])], dimensions=VIGNETTE_DIM)
            # vignetteS = extract_vignette(zscore_percentile(volume[int(centroids[k][0]), :, :]), [int(centroids[k][1]), int(centroids[k][2])], dimensions=VIGNETTE_DIM)
            vignetteC = extract_vignette(zscore_percentile(volume[:, int(centroids[k][1]), :]), [int(centroids[k][0]), int(centroids[k][2])], dimensions=VIGNETTE_DIM)
            
            # Make label prediction with the classification network
            vignette_pytorch = read_and_preprocess_multiview(vignetteA, vignetteC)
            # vignette_pytorch = read_and_preprocess_multiview(vignetteA, vignetteS)
            # vignette_pytorch = read_and_preprocess_multiview(vignetteS, vignetteC)

            with torch.no_grad():
                # model to eval mode
                model.eval()
                y_pred = model(vignette_pytorch)
                y_pred_tag = torch.softmax(y_pred, dim = 1)
                _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
                y_pred_tags = y_pred_tag[0][1] > cfg.inference.threshold

                if y_pred_tags == 0:
                    # Modify the prediction according to the classification output
                    lesion_cluster = np.where(output == k)
                    pred_mask_correction[lesion_cluster] = 0
                    stat0.append(y_pred[0][0].numpy())
                    stat0log.append(y_pred_tag[0][0].numpy())
                else:
                    stat1.append(y_pred[0][1].numpy())
                    stat1log.append(y_pred_tag[0][1].numpy())
    
    # Save new prediction (with correction)
    np.save(os.path.join(PATH_PAT, 'pred_correction_cactus_'+str(cfg.inference.threshold)+'.npy'), pred_mask_correction)

    #----------------------
    #    New Evaluation
    #----------------------

    new_scores = get_scores(patient, cfg.inference.threshold, gt_mask, pred_mask_correction)
    row_list.append(new_scores)


#####################
#  SAVE THE RESULTS #
#####################


df = pd.DataFrame(row_list, columns=['PatientID', 'Threshold', 'Dice', 'Dice TP',
    'Recall (Lesion-wise)', 'Precision (Lesion-wise)', 'F1-Score (Lesion-wise)',
    'VPP (Slice-wise)', 'VPN (Slice-wise)'])

df.to_csv('results/CACTUS_results.csv', index=False) # MODIFIER NOM
import numpy as np
import torch
import cc3d
import cv2

from torchvision import transforms
from sklearn.metrics import confusion_matrix

class EpochCallback:
    end_training = False
    not_improved_epoch = 0
    monitor_value = np.inf

    def __init__(self, model_name, total_epoch_num, model, optimizer, monitor=None, patience=None):
        if isinstance(model_name, str):
            model_name = [model_name]
            model = [model]
            optimizer = [optimizer]

        self.model_name = model_name
        self.total_epoch_num = total_epoch_num
        self.monitor = monitor
        self.patience = patience
        self.model = model
        self.optimizer = optimizer

    def __save_model(self):
        for m_name, m, opt in zip(self.model_name, self.model, self.optimizer):
            torch.save({'model_state_dict': m.state_dict(),
                        'optimizer_state_dict': opt.state_dict()},
                       m_name)

            print(f'Model saved to {m_name}')

    def epoch_end(self, epoch_num, hash):
        epoch_end_str = f'Epoch {epoch_num}/{self.total_epoch_num} - '
        for name, value in hash.items():
            epoch_end_str += f'{name}: {round(value, 4)} '

        print(epoch_end_str)

        if self.monitor is None:
            self.__save_model()

        elif hash[self.monitor] < self.monitor_value:
            print(f'{self.monitor} decreased from {round(self.monitor_value, 4)} to {round(hash[self.monitor], 4)}')

            self.not_improved_epoch = 0
            self.monitor_value = hash[self.monitor]
            self.__save_model()
        else:
            print(f'{self.monitor} did not decrease from {round(self.monitor_value, 4)}, model did not save!')

            self.not_improved_epoch += 1
            if self.patience is not None and self.not_improved_epoch >= self.patience:
                print("Training was stopped by callback!")
                self.end_training = True


#####################
#   PREPROCESSING   #
#####################


def extract_vignette(image, centroid, dimensions=(64, 64)):
    """
    Extract a vignette of given dimensions from an image
    centered around the centroid coordinates (connected component)
    """
    # print(centroid[1], centroid[0])
    x1 = int(centroid[0])-(dimensions[0]//2)
    x2 = int(centroid[0])+(dimensions[0]//2)
    y1 = int(centroid[1])-(dimensions[1]//2)
    y2 = int(centroid[1])+(dimensions[1]//2)
    # print(x1, x2, y1, y2)

    # Dimension verification
    if (x2-x1) != dimensions[0]:
        d = dimensions[0] - (x2-x1)
        x2 += d
    if (y2-y1) != dimensions[1]:
        d = dimensions[1] - (y2-y1)
        y2 += d

    # If vignette out of bounds
    if x1 < 0:
        # print(1)
        x2 -= x1
        x1 -= x1
    if x2 > image.shape[0]:
        # print(2)
        x1 += (image.shape[0])-x2
        x2 += (image.shape[0])-x2
    if y1 < 0:
        # print(3)
        y2 -= y1
        y1 -= y1
        
    if y2 > image.shape[1]:
        # print(4)
        y1 += (image.shape[1])-y2
        y2 += (image.shape[1])-y2
    
    vignette = image[x1:x2, y1:y2]
    coordinates = [[x1, x2], [y1, y2]]

    assert vignette.shape == dimensions

    return vignette


def zscore_percentile(image, pmin=2, pmax=98):

    volume = image.copy().astype(float)
    # print('[INFO] VIGNETTE TYPE AFTER NORM :', volume.dtype)
    volume[volume < np.nanpercentile(volume, pmin)] = np.nan
    volume[volume > np.nanpercentile(volume, pmax)] = np.nan

    z = (image - np.nanmean(volume)) / np.nanstd(volume)

    return z


def read_and_preprocess_multiview(imgA, imgC):

    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    imageA = imgA.astype(np.float32)
    imageC = imgC.astype(np.float32)

    # select axial and coronal projection & duplicate over 3 channels
    imageA = np.dstack((imageA, imageA, imageA)) # axial slice only
    imageC = np.dstack((imageC, imageC, imageC)) # coronal slice only
    
    # convert from float32 to uint8
    imageA = cv2.normalize(imageA, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    imageC = cv2.normalize(imageC, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    img_torchA = val_transforms(imageA)
    img_torchC = val_transforms(imageC)

    imageAC = [img_torchA, img_torchC]
    imageAC = torch.cat(imageAC, dim=0)
    img_torchAC = torch.unsqueeze(imageAC, 0)

    return img_torchAC


######################
#   POSTPROCESSING   #
######################


def remove_small_lesions(mask, percentage=0.9):
    """
    Only keep 90% of the biggest lesions from a segmentation map (3D Mask)
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    lc = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        return mask#, 0
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # Total number of lesion voxels
    lesion_total = np.sum(mask)
    # Initiate cumulated lesion volume
    volume = 0

    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # If the biggest lesion is larger than 90% of the lesional volume
        if lesions[k] >= lesion_total*percentage and c == 0 and N > 1:
            smallest_lesion_size = lesions[k]
            lesion_cluster = np.where(output == k) # correction Flora
            new_mask[lesion_cluster] = 1
            lc +=1
            c = 1

        volume += lesions[k]
        
        # Assure that we keep 90% of the biggest lesions by the end
        if volume <= lesion_total*percentage and N > 1:
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            
    # Condition to keep if only 1 lesion
    if N == 1:
        lesion_cluster = np.where(output == 1)
        new_mask[lesion_cluster] = 1
        lc += 1
    
    if N > 1:
        smallest_lesion_size = lesion_cluster[0].shape[0]
    else: # Only 1 lesion
        smallest_lesion_size = lesion_total

    print(lc, 'lesions kept in GT over', N, 'in total (min lesion size =', smallest_lesion_size)#, 'voxels or', round(vox2mm3(smallest_lesion_size), 2), 'mm3)')

    return new_mask


def remove_small_lesions_threshold(mask, min_lesion_size=10):
    """
    Only keep lesions bigger than `min_lesion_size` from a segmentation map (3D Mask)
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    lc = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        # print('No lesion found after crop.')
        return mask #, None, None
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # Total number of lesion voxels
    lesion_total = np.sum(mask)
    # Initiate cumulated lesion volume
    volume = 0
    
    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # Assure that we keep lesions bigger than threshold
        if lesions[k] > min_lesion_size:
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            volume += lesions[k]
            # print('keep lesion', k, volume)
            c = 1
    
    # if at least one lesion was above threshold
    if N > 1 and c == 1:
        smallest_lesion_size = lesion_cluster[0].shape[0]
    
    # if there is no lesion above the threshold
    if N > 1 and c == 0:
        print('NEW MASK WITH NO LESIONS ABOVE THRESHOLD :', np.max(new_mask))
        return new_mask #, None, None

    if N == 1: # Only 1 lesion
        smallest_lesion_size = lesion_total

    # print(lc, 'lesions kept over', N, 'in total (min lesion size =', smallest_lesion_size, 'voxels or', round(vox2mm3(smallest_lesion_size), 2), 'mm3)')
    # print(f'{round(volume/lesion_total*100, 1)}% of the lesional volume remaining.')
    return new_mask


def remove_large_lesions_threshold(mask, max_lesion_size=10000):
    """
    Only keep lesions bigger than `max_lesion_size` from a segmentation map (3D Mask)
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    lc = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        # print('No lesion found after crop.')
        return mask #, None, None
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # Total number of lesion voxels
    lesion_total = np.sum(mask)
    # Initiate cumulated lesion volume
    volume = 0
    
    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # Assure that we keep lesions bigger than threshold
        # print(k, vox2mm3(lesions[k]))
        if lesions[k] < max_lesion_size:
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            volume += lesions[k]
            # print('keep lesion', k, volume)
            c = 1

    return new_mask 


def remove_lesions_threshold(mask, min_lesion_size=3, max_lesion_size=100):
    """
    Only keep lesions bigger than `max_lesion_size` from a segmentation map (3D Mask)
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    lc = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        # print('No lesion found after crop.')
        return mask #, None, None
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # Total number of lesion voxels
    lesion_total = np.sum(mask)
    # Initiate cumulated lesion volume
    volume = 0
    
    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # Assure that we keep lesions bigger than threshold
        # print(k, vox2mm3(lesions[k]))
        if min_lesion_size <= lesions[k] :#<= max_lesion_size:
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            volume += lesions[k]
            # print('keep lesion', k, volume)
            c = 1
            
    print(f'{round(volume/lesion_total*100, 1)}% of the lesional volume remaining.')

    return new_mask 


############################
#   EVALUATION FUNCTIONS   #
############################


def extract_TP_FP(gt_mask, pred_mask, percentage=1):
    """
    Return True Positive and False Positive Masks from Prediction.
    """
    TP_mask = np.zeros(pred_mask.shape, dtype=np.uint8)
    FP_mask = np.zeros(pred_mask.shape, dtype=np.uint8)
    tp = 0
    fp = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(pred_mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesions remaining after classification.')
        return pred_mask, 0

    else: 
        # Get number of voxels, bounding box and centroid for each lesion
        stats = cc3d.statistics(output)

        # Number of voxels per lesion cluster with segid as key
        lesions = dict(enumerate(stats['voxel_counts']))
        # print(lesions)
        # First element is always the background
        del lesions[0]

        # Total number of lesion voxels
        lesion_total = np.sum(pred_mask)
        # Initiate cumulated lesion volume
        volume = 0

        c = 0
        for k in sorted(lesions, key=lesions.get, reverse=True):
            # If the biggest lesion is larger than 90% of the lesional volume
            if lesions[k] >= lesion_total*percentage and c == 0 and N > 1:
                smallest_lesion_size = lesions[k]
                lesion_cluster = np.where(output == k)
                # Check if false positive
                if np.max(gt_mask[lesion_cluster]) < 1: # False positive
                    # print(f'FALSE POSITIVE {k}')
                    FP_mask[lesion_cluster] = 1
                    fp += 1
                else:
                    # print(f'TRUE POSITIVE {k}') # True positive
                    TP_mask[lesion_cluster] = 1
                    tp += 1
                c = 1

            volume += lesions[k]
            
            # Assure that we keep 90% of the biggest lesions by the end
            if volume <= lesion_total*percentage and N > 1:
                lesion_cluster = np.where(output == k)
                # Check if false positive
                if np.max(gt_mask[lesion_cluster]) < 1: # False positive
                    # print(f'FALSE POSITIVE {k}')
                    FP_mask[lesion_cluster] = 1
                    fp += 1
                else:
                    # print(f'TRUE POSITIVE {k}') # True positive
                    TP_mask[lesion_cluster] = 1
                    tp += 1
                
        # Condition to keep if only 1 lesion
        if N == 1:
            lesion_cluster = np.where(output == 1)
            # Check if false positive
            if np.max(gt_mask[lesion_cluster]) < 1: # False positive
                # print(f'FALSE POSITIVE {k}')
                FP_mask[lesion_cluster] = 1
                fp += 1
            else:
                # print(f'TRUE POSITIVE {k}') # True positive
                TP_mask[lesion_cluster] = 1
                tp += 1
        
        if N > 1:
            smallest_lesion_size = lesion_cluster[0].shape[0]
        else: # Only 1 lesion
            smallest_lesion_size = lesion_total

        print(fp, 'false lesions and', tp, 'true lesions kept over', N, 'lesions remaining after classif in total.\n') #(min lesion size =', smallest_lesion_size, 'voxels or', round(vox2mm3(smallest_lesion_size), 2), 'mm3)')

        return TP_mask, FP_mask #, smallest_lesion_size


def recall_and_precision_lesion(y_true, y_pred, lesion_min=10, boundig_box_margin=15):

    recall = []
    precision = []

    for s in range(y_pred.shape[2]):
        # Select slice
        # print(s)
        # element_image = volume[:, :, s]
        element_mask = y_true[:, :, s].astype(np.uint8)
        element_prediction = y_pred[:, :, s].astype(np.uint8)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=element_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # number of true lesions in the groundtruth mask
        nb_lesions = len(contours)

        # initialize a list to store TRUE lesion detection and compute recall & precision afterwards
        lesions_detected = np.zeros(nb_lesions)
        # number of false positives
        FP = 0

        # apply connected component analysis to the thresholded image
        connectivity = 8
        output = cv2.connectedComponentsWithStats(element_prediction, connectivity, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # loop over the number of unique connected component labels, skipping
        # over the first label (as label zero is the background)
        for i in range(1, numLabels):
            # extract the connected component statistics for the current# label
            # x = stats[i, cv2.CC_STAT_LEFT]
            # y = stats[i, cv2.CC_STAT_TOP]
            # w = stats[i, cv2.CC_STAT_WIDTH]
            # h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            # print('\nConnected component', i, ':', area, 'pixels')
            # print('Barycenter coordinates :', (cX, cY))

            # If there is no lesion and the cc predicted is significative
            if len(contours)==0:
                # print('NO LESIONS')
                keepArea = area > lesion_min
                if keepArea:
                    FP += 1

            else:
                # Check if the current CC is a TRUE POSITIVE
                cc = 0
                
                for j in range(len(contours)):
                    # ensure the width, height, and area are all neither too small
                    # nor too big
                    # keepWidth = w > 5 and w < 50
                    # keepHeight = h > 45 and h < 65
                    keepArea = area > lesion_min
                    # ensure the connected component we are
                    # examining passes all three tests
                    if keepArea:
                        # construct a mask for the current connected component and
                        # then take the bitwise OR with the mask
                        # print("[INFO] keeping connected component '{}'".format(i))
                        # print("[CONTOUR "+str(j)+"]")

                        # Check if barycenter belong to the polygonal contour of TRUE lesion
                        result = cv2.pointPolygonTest(contours[j], (cX, cY), False)
                        # print(result)
                        # Check if barycenter belong to the rectangular contour of TRUE lesion
                        x,y,w,h = cv2.boundingRect(contours[j]) # BOUNDING BOX
                        x -= boundig_box_margin
                        y -= boundig_box_margin
                        w += boundig_box_margin
                        h += boundig_box_margin
                        res = x <= cX < x+w and y <= cY < y + h
                        # print(res)
                        if res:
                            lesions_detected[j] = 1
                            cc = 1
                    else:
                        # print("[INFO] removing connected component '{}'".format(i))
                        # we ignore the lesion for computing recall and precision
                        cc = 1

                # If cc DO NOT belong to any of the TRUE lesions
                if cc == 0:
                    FP += 1

        # if no lesions in GT AND correct prediction (no lesion in the predicted mask)
        # print(nb_lesions, FP, recall, precision)
        if nb_lesions==0:
            if FP == 0:
                precision.append(1)
            if FP > 0: # FP > 0
                precision.append(0)
        
        else:
            recall.append(np.sum(lesions_detected)/len(lesions_detected))
            if (np.sum(lesions_detected) + FP) > 0:
                precision.append(np.sum(lesions_detected)/(np.sum(lesions_detected) + FP))
            else:
                precision.append(0)

    f1score = 2 * (np.mean(recall) * np.mean(precision)) / (np.mean(recall) + np.mean(precision))
    return np.mean(recall), np.mean(precision), f1score


def dice_coef2(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    intersection = np.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (union + smooth)
        

def confusion_matrix_slice(y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    FN_pred = []
    TP_pred = []

    for i in range(y_pred_f.shape[2]):
        cm_pw = confusion_matrix(y_true_f[:, :, i].flatten(), y_pred_f[:, :, i].flatten())
        if np.max(y_pred_f[:, :, i]) == 1 and np.max(y_true_f[:, :, i]) == 1:
            # We check if the prediction intersect with the groundtruth
            if cm_pw[1][1] > 0:
                TP += 1
                TP_pred.append(i)
            else:
                FN += 1
                FN_pred.append(i)
            # TP += 1
        elif np.max(y_pred_f[:, :, i]) == 1 and np.max(y_true_f[:, :, i]) == 0:
            FP += 1
        elif np.max(y_pred_f[:, :, i]) == 0 and np.max(y_true_f[:, :, i]) == 0:
            TN += 1
        else:
            FN += 1

    return TP, FP, TN, FN, FN_pred, TP_pred


def get_scores(patient, threshold, y_true, y_pred, decimal=4):
    scores = {'PatientID': patient, 'Threshold': threshold}

    # # IoU
    # scores['IoU_b'] = np.round(jaccard_score(y_true.flatten(), y_pred.flatten(), average='binary'), decimal)
    # scores['IoU_m'] = np.round(jaccard_score(y_true.flatten(), y_pred.flatten(), average='macro'), decimal)
    # scores['IoU_w'] = np.round(jaccard_score(y_true.flatten(), y_pred.flatten(), average='weighted'), decimal)

    # Dice
    scores['Dice'] = np.round(dice_coef2(y_true, y_pred), decimal)

    # Dice computed over TP ONLY
    TP, _ = extract_TP_FP(y_true, y_pred, percentage=1)
    scores['Dice TP'] = np.round(dice_coef2(y_true, TP), decimal)


    # Recall & Precision (Lesion-wise)
    recall, precision, f1 = recall_and_precision_lesion(y_true, y_pred, lesion_min=10, boundig_box_margin=15) #15
    scores['Recall (Lesion-wise)'] = np.round(recall, decimal)
    scores['Precision (Lesion-wise)'] = np.round(precision, decimal)
    scores['F1-Score (Lesion-wise)'] = np.round(f1, decimal)

    # # Confusion matrix (pixel-wise)
    # cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    # # try:
    # #     scores['True Intersection (Pixel-wise)'] = np.round(cm[1] / np.sum(cm[1]), decimal)[1]
    # # except RuntimeWarning:
    # if cm.shape != (1, 1):
    #     scores['True Intersection (Pixel-wise)'] = np.round(cm[1] / np.sum(cm[1]), decimal)[1]
    #     scores['FN (Pixel-wise)'] = cm[1][0]
    #     scores['TP (Pixel-wise)'] = cm[1][1]
    # else:
    #     scores['True Intersection (Pixel-wise)'] = None
    #     scores['FN (Pixel-wise)'] = None
    #     scores['TP (Pixel-wise)'] = None

    # Confusion Matrix (slice-wise)
    TP, FP, TN, FN, FN_pred, TP_pred = confusion_matrix_slice(y_true, y_pred)

    try:
    #     # MCC
    #     scores['MCC (Slice-wise)'] = np.round((TP*TN-FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)), decimal)
    #     # Recall
    #     scores['Recall (Slice-wise)'] = np.round(TP/(TP+FN), decimal)

    #     # Precision
    #     scores['Precision (Slice-wise)'] = np.round(TP/(TP+FP), decimal)

    #     # F1-Score
    #     scores['F1-Score (Slice-wise)'] = np.round(TP / (TP + (1/2) * (FP + FN)), decimal)

    #     # Accuracy
    #     scores['Accuracy (Slice-wise)'] = np.round((TP+TN)/(TP+FP+FN+TN), decimal)

        # Valeur prédictive positive
        import math
        if math.isnan(recall):
            scores['VPP (Slice-wise)'] = None
            
        else:
            scores['VPP (Slice-wise)'] = np.round(TP/(TP+FP), decimal)
            

        # Valeur prédictive négative 
        scores['VPN (Slice-wise)'] = np.round(TN/(TN+FN), decimal)

    except ZeroDivisionError:
    #     scores['MCC (Slice-wise)'] = None
    #     scores['Recall (Slice-wise)'] = None
    #     scores['Accuracy (Slice-wise)'] = None
    #     scores['F1-Score (Slice-wise)'] = None
    #     scores['Dice'] = None
        scores['VPP (Slice-wise)'] = None
        scores['VPN (Slice-wise)'] = None

    # scores['TN'] = TN
    # scores['FP'] = FP
    # scores['FN'] = FN
    # scores['TP'] = TP
    # scores['FN_pred'] = FN_pred
    # scores['TP_pred'] = TP_pred

    return scores



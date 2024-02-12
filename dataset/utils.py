#####################
# IMPORT LIBRAIRIES #
#####################

import numpy as np
import cv2


#######################################
#    CONNECTED COMPONENT EXTRACTION   #
#######################################


def get_centroids(mask, cc_area_min=20):
    """
    Return centroids coordinates from connected components bigger than cc_area_min (in pixels)
    """
    c = []
    # apply connected component analysis to the thresholded image
    connectivity = 8
    output = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    
    # loop over the number of unique connected component labels, skipping
    # over the first label (as label zero is the background)
    for i in range(1, numLabels):
        # extract the connected component statistics for the current# label
        area = stats[i, cv2.CC_STAT_AREA]
        # ensure area is not too small
        keepArea = area > cc_area_min
        if keepArea:
            # add centroid to the list
            c.append(centroids[i])
            # print("[INFO] keeping connected component '{}'".format(i))
    
    return c


def zscore_percentile(image, pmin=2, pmax=98):
    """
    Return a volume with standardized values.
    """
    volume = image.copy().astype(float)
    volume[volume < np.nanpercentile(volume, pmin)] = np.nan
    volume[volume > np.nanpercentile(volume, pmax)] = np.nan

    z = (image - np.nanmean(volume)) / np.nanstd(volume)

    return z


def extract_vignette(image, centroid, dimensions=(64, 64)):
    """
    Extract a vignette of given dimensions from an image
    centered around the centroid coordinates (connected component)
    """

    x1 = int(centroid[0])-(dimensions[0]//2)
    x2 = int(centroid[0])+(dimensions[0]//2)
    y1 = int(centroid[1])-(dimensions[1]//2)
    y2 = int(centroid[1])+(dimensions[1]//2)

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
 
    assert vignette.shape == dimensions

    return vignette
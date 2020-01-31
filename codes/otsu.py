import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data.dataset import Dataset
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys, re
import math
import datetime
from skimage import measure
import time, datetime
from random import sample
from torch.autograd import Variable
from multiprocessing import Pool
from skimage.restoration import inpaint
import scipy.ndimage.filters as fi

#fonctions 
from codes.imgtotensor_patches_samples_two import ImageDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from models.AE_fully_convolutional_model import Encoder, Decoder
from codes.plot_loss import plotting
from codes.img_preprocessing import normalize

from random import sample
from skimage import filters
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import scipy.ndimage.filters as fi

gpu = torch.cuda.is_available()

# This code is taken from skimage
def histogram(image, nbins=256):
    # For integer types, histogramming with bincount is more efficient.
    if np.issubdtype(image.dtype, np.integer):
        offset = 0
        image_min = np.min(image)
        if image_min < 0:
            offset = image_min
            image_range = np.max(image).astype(np.int64) - image_min
            # get smallest dtype that can hold both minimum and offset maximum
            offset_dtype = np.promote_types(np.min_scalar_type(image_range),
                                            np.min_scalar_type(image_min))
            if image.dtype != offset_dtype:
                # prevent overflow errors when offsetting
                image = image.astype(offset_dtype)
            image = image - offset
        hist = np.bincount(image)
        bin_centers = np.arange(len(hist)) + offset

        # clip histogram to start with a non-zero bin
        idx = np.nonzero(hist)[0][0]
        return hist[idx:], bin_centers[idx:]
    else:
        hist, bin_edges = np.histogram(image.flat, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        return hist, bin_centers


# This code is taken from skimage
def threshold_otsu(image, nbins=256):
    hist, bin_centers = histogram(image, nbins)
    hist = hist.astype(float)
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def otsu(image_array_loss1,image_array_loss2, H, W,date1,date2,parameters,paths,loss_arrays,loss_arrays_otsu,changes=None):

    patch_size = parameters.get("patch_size")
    epoch_nb = parameters.get("epoch_nb")
    batch_size = parameters.get("batch_size")
    learning_rate = parameters.get("learning_rate")
    weighted = parameters.get("weighted")
    sigma = parameters.get("sigma")
    shuffle = parameters.get("shuffle")
    sampleTrue = parameters.get("sampleTrue")
    threshold = parameters.get("threshold")
    
    patient = paths.get("patient")
    path_results = paths.get("path_results")
    path_models = paths.get("path_models")
    path_models_finetuned = paths.get("path_models_finetuned")
    path_diff = paths.get("path_diff")
    path_bcm = paths.get("path_bcm")
    path_loss = paths.get("path_loss")
    path_reconstructed = paths.get("path_reconstructed")

    # We calculate the average reconstruction error image
    image_array_loss = np.divide((image_array_loss1+image_array_loss2), 2)
    #normalize loss for visual
    image_array_loss = normalize(image_array_loss)*255
#     for loss and diff comparaison
    loss_arrays.append(image_array_loss)
    
    # We rescale the image values to 8 bits so it works with the functions from skimage
#     max_ = np.nanmax(image_array_loss)
#     coef = max_/256
#     image_array_loss = image_array_loss/coef
#     image_array_loss = np.asarray(image_array_loss, dtype=int)

    # THIS IS VERY IMPORTANT VALUE
    # Otsu threshold is automatic, however before applying it, we exclude 0.5% of the highest reconstruction error values as they ae considered to be outliers
    # This parameter can be modified if needed
#     threshold = 1
#     threshold = 0.995
    # image_array_loss[~np.isnan(image_array_loss)].flatten() car on fait le threshold sans les nan
    a=image_array_loss[~np.isnan(image_array_loss)].flatten()
    
    val = filters.threshold_otsu(np.sort(a)[0:int(a.shape[0]*threshold)]) # Obtained threshold value
    
    del a

    # We get binary change map (1 - changes, 0 - no changes) using the threshold and write it to tiff and shp
    image_array_outliers = np.zeros(H*W)
    image_array_outliers[image_array_loss.flatten() > val] = 1
    outliers_image_mean = path_bcm+"BCM_"+date1+'_to_'+date2+"_threshold_" +str(threshold)+"_otsu_"+str(val)+'.png'
    
    image_array_outliers = np.reshape(image_array_outliers, (H, W))
    image_array_outliers=image_array_outliers.astype('uint8')
    image_array_outliers=image_array_outliers*255
#     image_gray = cv2.cvtColor(image_array_outliers, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(outliers_image_mean,image_array_outliers)
    
    loss_arrays_otsu.append([image_array_outliers,val])

    
    # We calculate the stats if the ground truth is available for this couple of images
    # if changes is not None:
    #     # path of ground truth image, I have only 2 GT
    #     path_cm = '/home/user/Dropbox/IJCNN/images/' + changes
    #     cm_truth_name = "mask_changes_small1"
    #     if changes=="changes_2004_2005":
    #         cm_predicted = (np.reshape(image_array_outliers, (H, W))[0:600, 600:1400]).flatten()
    #     if changes == "changes_2006_2008":
    #         cm_predicted = (np.reshape(image_array_outliers, (H, W))[100:370, 1000:1320]).flatten()

    #     cm_truth, _, _, _, _, _ = open_tiff(path_cm, cm_truth_name)
    #     cm_truth = cm_truth.flatten()
    #     cm_truth[cm_truth==255]=0
    #     #Different stats takenthreshold = 0.995 from scikit
    #     print(classification_report(cm_truth, cm_predicted, target_names=["no changes", "changes"]))
    #     print(accuracy_score(cm_truth, cm_predicted))
    #     print(cohen_kappa_score(cm_truth, cm_predicted))
    #     conf = confusion_matrix(cm_truth, cm_predicted)
    #     print(confusion_matrix(cm_truth, cm_predicted))
    #     omission = conf[1][0]/sum(conf[1])
    #     print (omission)
    
    return image_array_outliers
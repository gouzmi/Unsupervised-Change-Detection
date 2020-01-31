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

gpu = torch.cuda.is_available()

def normalize(x,min=None,max=None):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    if min==None:
        min=np.nanmin(x)
    if max==None:
        max=np.nanmax(x)
        
    return np.array((x - min) / (max - min))

def bouding_box(tab):
    
    mask = np.where(tab==0,tab,np.NaN)
    
    if(tab.shape==mask.shape):
        y_min = x_min = 0
        y_max = mask.shape[0]-1
        x_max = mask.shape[1]-1

        while(np.isnan(mask[y_min]).any()==False):
            y_min+=1

        while(np.isnan(mask[y_max]).any()==False):
            y_max-=1

        while(np.isnan(mask[:,x_min]).any()==False):
            x_min+=1

        while(np.isnan(mask[:,x_max]).any()==False):
            x_max-=1

        tab = tab[y_min:y_max,x_min:x_max]
        tab = np.where(tab==0,np.NaN, tab)
        
        return tab
    else:
        return "not same shape"

def gkern2(kernlen, sigma):
    """Returns a 2D Gaussian kernel array."""
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, sigma)*100

def extend(band, patch_size):
    to_insert1 = np.flipud(band[1: int(patch_size / 2 + 1)])
    band = np.insert(band, 0, to_insert1, axis=0)
    to_insert2 = np.flipud(band[(len(band) - int(patch_size / 2) - 1):(len(band) - 1)])
    band = np.concatenate((band, to_insert2), axis=0)
    to_insert3 = band[:, range(int(patch_size / 2), 0, -1)]
    band = np.concatenate((to_insert3, band), axis=1)
    to_insert4 = band[:, range((len(band[0]) - 2), (len(band[0]) - int(patch_size / 2) - 2), -1)]
    band = np.concatenate((band, to_insert4), axis=1)
    return band

def process_image(extend_img):

        print('yes')
        mask = np.where(np.isnan(extend_img),255,0)
        img_test = np.where(np.isnan(extend_img),0,extend_img)
        extend_img = inpaint.inpaint_biharmonic(img_test, mask, multichannel=False)
        print('done')
        return extend_img


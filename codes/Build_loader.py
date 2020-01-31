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
from codes.imgtotensor_patches_samples_list import ImageDataset
from torch.utils.data import DataLoader

gpu = torch.cuda.is_available()

def dsloader(image, batch_size, shuffle):
    if gpu:
        loader = DataLoader(image,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=1,  # 1 for CUDA
                            pin_memory=True,  # CUDA only
                            drop_last=False
                            )
    else:
        loader= None
    return loader

def build_loader(imgs,extend_imgs,parameters):

    patch_size = parameters.get("patch_size")
    batch_size = parameters.get("batch_size")

    flat = imgs[0].flatten()
    index_not_nan = np.where(np.isnan(flat) == False)
    index_not_nan = index_not_nan[0]
    
    
    image = None    # Dataset with the sample of patches from all images
    nb_pixels_not_nan = np.count_nonzero(~np.isnan(imgs[0]))
    nbr_patches_per_image = int(nb_pixels_not_nan/len(imgs))   # We sample H*W/ number of images patches from every image
    # nbr_patches_per_image = int(nb_pixels_not_nan/2) 

    for ii in range(len(extend_imgs)):

        samples_list = np.sort(sample(list(index_not_nan), nbr_patches_per_image)) #pixels tirés au hasard
        if image is None:
            image = ImageDataset(extend_imgs[ii], patch_size, ii,
                                 samples_list)  # we create a dataset with tensor patches
        else:
            image2 = ImageDataset(extend_imgs[ii], patch_size, ii,
                                  samples_list)  # we create a dataset with tensor patches
            image = torch.utils.data.ConcatDataset([image, image2])
    loader = dsloader(image, batch_size, shuffle=True) # dataloader
    
    print(10*('-'))
    print("loader created")
    print(10*('-'))
    return loader
    
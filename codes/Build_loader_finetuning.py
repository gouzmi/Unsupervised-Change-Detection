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

from random import sample
from skimage import filters
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import scipy.ndimage.filters as fi

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

def build_loader(imgs,extend_imgs,patch_size,batch_size,shuffle):

    flat = imgs[0].flatten()
    index_not_nan = np.where(np.isnan(flat) == False)
    index_not_nan = index_not_nan[0]

    image = ImageDataset(extend_imgs[0], extend_imgs[1], patch_size,list(index_not_nan))  # we create a dataset with tensor patches
    loader = dsloader(image, batch_size, shuffle)
    print(10*('-'))
    print("loader created")
    print(10*('-'))
    return loader
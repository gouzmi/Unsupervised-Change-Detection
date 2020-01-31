import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys, re
import math
from skimage import measure
import time, datetime
from codes.img_preprocessing import *

def composition(path_bcm,pop_first=True,overlap=False):

    #     get imgs to score ordered by time
    os.chdir(path_bcm)
    files = filter(os.path.isfile, os.listdir(path_bcm))
    files = [os.path.join(path_bcm, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    files = [cv2.imread(f,0) for f in files]
    if pop_first:
        files.pop(0)
    
    inter = (255/len(files))

    composition = np.zeros(files[0].shape)

    for i in range(len(files)):
        img = np.where(files[i]!=0,(i+1)*int(inter),0)

        if overlap:
            #overlap
            composition += img
            composition = np.where(composition>255,255,composition)
        else:
            #no overlap (si deux pixels sont marqués, la couleur représentera le changement survenu le +tôt)
            composition = np.where(composition==0,img,composition) 

         

    RB = np.where(composition==0,1,0)
    composition = normalize(composition)
    composition = np.where(composition==0,1,composition)
    composition = np.stack((RB,composition, RB), axis=2)

    cv2.imwrite(path_bcm[:-3]+'composition.png',composition*255)
    plt.imshow(composition)

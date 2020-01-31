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
from codes.imgtotensor_patches_samples_list import ImageDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from models.AE_fully_convolutional_model import Encoder, Decoder
from codes.plot_loss import plotting
from codes.Build_loader import *
from codes.img_preprocessing import *
from codes.pretrain import pretrain

#parameters
#-------------------------------------------------------------------------
patch_sizes = [5,7]
epoch_nb = 8
batch_size = 100
learning_rate = 0.0001
weighted = True            # if we weight patches loss (center pixel has higher loss)
sigmas = [5,7,9]                   # sigma for weighted lossloader = build_loader(extend_imgs,H,W)
shuffle = False             # shuffle patches before training

#patient = '102_SE_J/IR OD/'
patient ='005_AU_R/IR OD/'
# patient ='001_AF_S/IR OD/spe/'
# patient = '003_AL_J/IR OD/'
# patient ='008_BA_D/IR OG/spe/'
# patient = '010_BA_J/IR OG/'

gpu = torch.cuda.is_available()


#images recuperation
#-------------------------------------------------------------------------

imgs=[]
racine = '/media/guillaume/OS/Users/gouzm/Documents/DMLA-TimeLapse-Align-corrected/'
for subdir, dirs, files in os.walk(racine+patient):
    for file in files:
        imgs.append(racine+patient+ file)

print("nombres d'images : ",len(imgs))

#-------------------------------------------------------------------------
#min bouding box
cropped_imgs = []
for i in range(len(imgs)):
    img = bouding_box((cv2.imread(imgs[i],0)))
    cropped_imgs.append(img)

#normalisation de la série entre 0 et 1 pour les réseaux
cropped_imgs = normalize(cropped_imgs)

#-------------------------------------------------------------------------
def extend_img(cropped_imgs,patch_size):
    H,W = cropped_imgs[0].shape
    extend_imgs = None
    for im in range(len(cropped_imgs)):
        extended = np.reshape(extend(cropped_imgs[im],patch_size), (-1, H+patch_size-1, W+patch_size-1))
        if extend_imgs is None:
            extend_imgs = extended
        else:
            extend_imgs = np.concatenate((extend_imgs, extended), axis=0)
            
    t1 = time.time()

    if __name__ == '__main__':

        pool = Pool()                         # Create a multiprocessing Pool
    #     pool = Pool(os.cpu_count())
    #     pool = Pool(os.cpu_count()-1)
        out = pool.map(process_image, extend_imgs)

        pool.close()
        pool.join()

    print("--- %s seconds ---" % (time.time() - t1))
            
    return out

for patch_size in patch_sizes:
    for sigma in sigmas:

        print(100*('-'))
        print('Pretraining with patch_size : ',patch_size)
        print(100*('-'))

        parameters = {
        "patch_size": patch_size,
        "epoch_nb": epoch_nb,
        "batch_size": batch_size,
        "learning_rate":learning_rate,
        "weighted":weighted,  
        "sigma":sigma,         
        "shuffle":shuffle}

        if weighted:
            path_results = '/home/guillaume/Documents/Env_DMLA/results/'+patient+'patch_size_'+str(patch_size)+'/weighted/sigma'+str(sigma)+'/'
        else:
            path_results = '/home/guillaume/Documents/Env_DMLA/results/'+patient+'patch_size_'+str(patch_size)+'/normal/'

        path_imgs = path_results+'img_results/'
        path_models = path_results+'models/'
        
        paths = {
        "patient" : patient,
        "path_results" : path_results,
        "path_imgs" : path_imgs,
        "path_models" : path_models}
        
        extend_imgs = extend_img(cropped_imgs,patch_size)

        #-------------------------------------------------------------------------
        #affichage des images

        # fig = plt.figure(figsize=(20, 30))
        # columns = 3
        # rows = math.ceil(len(extend_imgs)/columns)
        # # ax enables access to manipulate each of subplots
        # ax = []

        # for i in range(len(imgs)):
        #     # create subplot and append to ax
        #     ax.append( fig.add_subplot(rows, columns, i+1) )
        #     ax[-1].set_title("ax:"+str(i))  # set title
            
        #     plt.imshow(extend_imgs[i],cmap='gray')

        # plt.show()

        #-------------------------------------------------------------------------
        # Data shaping & pretraining

        t1 = time.clock()

        # print('Processing : ',i,i+2)

        # date1=(imgs[i].replace(racine+patient,''))  
        # date2=(imgs[i+1].replace(racine+patient,''))  
        # date1=(re.search(r'[2][0]\d{6}', date1)).group(0)
        # date2=(re.search(r'[2][0]\d{6}', date2)).group(0)
        # print(date1,'to',date2)
        # pairs_img = cropped_imgs[i:i+2]
        # pairs_extended = extend_imgs[i:i+2]


        if not os.path.exists(path_results):
                os.makedirs(path_results)

        print(100*'=')
        param = "Patch_size = {} | Epoch_nb = {} | Batch_size = {} | Learning_rate = {}".format(patch_size,epoch_nb,batch_size,learning_rate)
        print(param)

        loader = build_loader(cropped_imgs,extend_imgs,parameters) #training dataset

        # We create AE model
        encoder = Encoder(1, patch_size) 
        decoder = Decoder(1, patch_size) 
        if gpu:
            print('On GPU :)')
            print(10*('-'))
            encoder = encoder.to('cuda:0')  
            decoder = decoder.to('cuda:0') 
        else:
            print('On CPU :(')
            print(10*('-'))

            
        #pretrain
        best_epoch_loss = pretrain(encoder,decoder,loader,parameters,paths)
        print(100*'=')


        t2 = time.clock()
        total_time_learning = t2 - t1
        total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
        print("Total time learning =", total_time_learning)

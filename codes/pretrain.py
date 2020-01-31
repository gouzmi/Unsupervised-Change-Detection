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
from codes.img_preprocessing import gkern2

gpu = torch.cuda.is_available()

def pretrain(encoder,decoder,loader,parameters,paths):

    patch_size = parameters.get("patch_size")
    epoch_nb = parameters.get("epoch_nb")
    batch_size = parameters.get("batch_size")
    learning_rate = parameters.get("learning_rate")
    weighted = parameters.get("weighted")
    sigma = parameters.get("sigma")
    shuffle = parameters.get("shuffle")
    
    patient = paths.get("patient")
    path_results = paths.get("path_results")
    path_imgs = paths.get("path_imgs")
    path_models = paths.get("path_models")
    
    start_time = time.clock()
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate)    #optimizer
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()    # loss function
    
    #create dir for the future model
    if not os.path.exists(path_models):
        os.makedirs(path_models)

    # Function to pretrain the model, pretty much standart
    epoch_loss_list = []
    for epoch in range(epoch_nb):
        encoder.train()
        decoder.train()
        total_loss = 0
        for batch_idx, (data, _, _) in enumerate(loader):
            if gpu:
                data = data.cuda(async=True)

            encoded = encoder(Variable(data))
            decoded = decoder(encoded)

            # we calculate batch loss to optimize the model
            #             if weighted
            if weighted:
#                 creation des filtres gaussiens
                weight = torch.from_numpy(gkern2(patch_size, sigma)).float().expand(decoded.shape[0], 1, patch_size, patch_size)
                if gpu:
                    weight = weight.cuda()
                
                loss = criterion(decoded*Variable(weight), Variable(data)*Variable(weight))
            
            else:
                loss = criterion(decoded, Variable(data))

            total_loss += loss.item()
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()
            if (batch_idx+1) % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                    (epoch+1), (batch_idx+1) * batch_size, len(loader.dataset),
                    100. * (batch_idx+1) / len(loader), loss.item()))
        epoch_loss = total_loss / len(loader)
        epoch_loss_list.append(epoch_loss)
        epoch_stats = "Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch + 1, epoch_loss)
        print(epoch_stats)
        with open(path_models + "stats.txt", 'a') as f:
            f.write(epoch_stats+"\n")
        f.close()
        
        
       
    run_name = "."+str(time.strftime("%Y-%m-%d_%H%M"))

    # we save the model
    torch.save([encoder, decoder], (path_models+'_run_'+str(run_name)+'nb_epoch'+str(epoch_nb)+"_loss_"+str(round(epoch_loss, 7))+'.pkl') )
    
    #save loss plot
    plotting(epoch_nb, epoch_loss_list,patch_size,batch_size,learning_rate, path_imgs,run_name)
    
    # Some stats about the best epoch loss and learning time
    best_epoch = np.argmin(np.asarray(epoch_loss_list))+1
    best_epoch_loss = epoch_loss_list[best_epoch-1]


    print("best epoch " + str(best_epoch))
    print("best epoch loss " + str(best_epoch_loss))

    end_time = time.clock()
    total_time_learning = end_time - start_time
    total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
    print("Total time learning for this pair =", total_time_learning)
    
    return best_epoch_loss
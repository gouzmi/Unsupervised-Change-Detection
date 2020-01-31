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
from codes.img_preprocessing import gkern2

from random import sample
from skimage import filters
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import scipy.ndimage.filters as fi

gpu = torch.cuda.is_available()

def train(loader,encoder12,decoder12,encoder21,decoder21,date1,date2,parameters,paths):

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

    epoch_loss_list = []
    epoch_loss12_list = []
    epoch_loss21_list = []
    run_names = []
        
    start_time = time.clock()
    
    optimizer = torch.optim.Adam((list(encoder12.parameters()) + list(decoder12.parameters()) + list(encoder21.parameters()) + list(decoder21.parameters())), lr=learning_rate)
    criterion = nn.MSELoss()    #loss function

    
    for epoch in range(epoch_nb):
        
        encoder12.train() #we swich to train mode (by default)
        decoder12.train()
        encoder21.train() #we swich to train mode (by default)
        decoder21.train()
        total_loss = 0
        total_loss12 = 0
        total_loss21 = 0
        
        for batch_idx, (data1, data2, _) in enumerate(loader):
            if gpu:
                data1 = data1.cuda(async=True)
                data2 = data2.cuda(async=True)

            encoded12 = encoder12(Variable(data1))
            encoded21 = encoder21(Variable(data2))
            decoded12 = decoder12(encoded12)
            decoded21 = decoder21(encoded21)
            
            # We calculate the loss of the bottleneck
            encoded21_copy = encoded21.clone().detach()
            encoded12_copy = encoded12.clone().detach()
            
                        # we calculate batch loss to optimize the model
            #             if weighted
            if weighted:
#                 creation des filtres gaussiens
                weight = torch.from_numpy(gkern2(patch_size, sigma)).float().expand(decoded12.shape[0], 1, patch_size, patch_size)
                if gpu:
                    weight = weight.cuda()
                
                #             bottleneck loss
                loss11 = criterion(encoded12*Variable(weight), Variable(weight)*(encoded12_copy+encoded21_copy)/2)
                loss22 = criterion(encoded21*Variable(weight), Variable(weight)*(encoded12_copy+encoded21_copy)/2)

                # We calculate the reconstruction loss for two AEs
                loss12 = criterion(decoded12*Variable(weight), Variable(data2)*Variable(weight))
                loss21 = criterion(decoded21*Variable(weight), Variable(data1)*Variable(weight))
            
            else:
                #             bottleneck loss
                loss11 = criterion(encoded12, (encoded12_copy+encoded21_copy)/2)
                loss22 = criterion(encoded21, (encoded12_copy+encoded21_copy)/2)

                # We calculate the reconstruction loss for two AEs
                loss12 = criterion(decoded12, Variable(data2))
                loss21 = criterion(decoded21, Variable(data1))
                

            
            total_loss += loss11.item()             #total loss for the epoch
            total_loss12 += loss12.item()
            total_loss21 += loss21.item()
            
            optimizer.zero_grad()           #everything to optimize the model
            loss11.backward(retain_graph=True)  # retain_graph is used to deal with complex model with many branches
            loss22.backward(retain_graph=True)
            loss12.backward(retain_graph=True)
            loss21.backward()
            optimizer.step()
        
            
            if (batch_idx+1) % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}\tLoss12: {:.7f}\tLoss21: {:.7f}'.format(
                    (epoch+1), (batch_idx+1) * batch_size, len(loader.dataset),
                    100. * (batch_idx+1) / len(loader), loss11.item(), loss12.item(), loss21.item()))
        
        epoch_loss = total_loss / len(loader)   #avg loss per epoch
        epoch_loss_list.append(epoch_loss)
        epoch_loss12 = total_loss12 / len(loader)   #avg loss per epoch
        epoch_loss12_list.append(epoch_loss12)
        epoch_loss21 = total_loss21 / len(loader)   #avg loss per epoch
        epoch_loss21_list.append(epoch_loss21)
        epoch_stats = "Epoch {} Complete: Avg. Loss: {:.7f}\tAvg. Loss12: {:.7f}\tAvg. Loss21: {:.7f}".format(epoch + 1, epoch_loss, epoch_loss12, epoch_loss21)
        print(epoch_stats)
        with open(path_models_finetuned + "stats_"+date1+"_to_"+date2+".txt", 'a') as f:
            f.write(epoch_stats+"\n")
        f.close()
        
        run_name = "."+str(time.strftime("%Y-%m-%d_%H%M"))
        run_names.append(run_name)
        #we save all the models to choose the best afterwards
        torch.save([encoder12, decoder12], (path_models_finetuned+'id_'+date1+'_to_'+date2+'_ae12-model_ep_'+str(epoch)+"_loss_"+str(round(epoch_loss12, 7))+"_run_"+run_name+'.pkl') )
        torch.save([encoder21, decoder21], (path_models_finetuned+'id_'+date1+'_to_'+date2+'_ae21-model_ep_'+str(epoch)+"_loss_"+str(round(epoch_loss21, 7))+"_run_"+run_name+'.pkl') )

#     # Some stats about the best epoch loss and learning time
#     best_epoch = np.argmin(np.asarray(epoch_loss_list))+1
#     best_epoch_loss = epoch_loss_list[best_epoch-1]

# il faut ici dire quelle epoch eest la meilleure pour chaque modele

#     print("best epoch " + str(best_epoch))
#     print("best epoch loss " + str(best_epoch_loss))

    end_time = time.clock()
    total_time_learning = end_time - start_time
    total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
    print("Total time learning =", total_time_learning)
    
    return epoch_loss_list, epoch_loss12_list, epoch_loss21_list, run_names

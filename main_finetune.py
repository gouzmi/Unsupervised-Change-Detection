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
from skimage import filters
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

#fonctions 
from codes.imgtotensor_patches_samples_two import ImageDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from models.AE_fully_convolutional_model import Encoder, Decoder
from codes.plot_loss import plotting
from codes.img_preprocessing import *
from codes.Build_loader_finetuning import dsloader, build_loader
from codes.train import train
from codes.otsu import histogram,otsu,threshold_otsu
from codes.score import score
import natsort

#parameters
#-------------------------------------------------------------------------

patch_sizes = [7,13]
epoch_nb = 1
batch_size = 100
learning_rate = 0.0001
weighted = True    # if we weight patches loss (center pixel has higher loss)
sigmas = [5,12]       # sigma for weighted loss
shuffle = True      # shuffle patches before training
sampleTrue = False  # if we train the model with all the patches or only with samles
threshold = 1

#patient = '102_SE_J/IR OD/'
# patient ='005_AU_R/IR OD/'
# patient ='001_AF_S/IR OD/spe/'
# patient = '003_AL_J/IR OD/'
# patient ='008_BA_D/IR OG/spe/'
# patient = '010_BA_J/IR OD/'
# patient ='018_BR_R/IR OD/'
# patient ='112_WA_P/IR OD/'
# patient = '016_BL_G/IR OG/'
# patient = '019_BU_C/IR OD/'
# patient = '089_PO_E/IR OD/'

patients =['019_BU_C/IR OD/','089_PO_E/IR OD/']
gpu = torch.cuda.is_available()

for patient in patients:

    print("---"+patient+"---")
    #to make score
    make_score = True
    path_original = '/home/guillaume/Documents/Env_DMLA/patients/'+patient
    path_true = '/home/guillaume/Documents/Env_DMLA/patients GT/'+patient

    gpu = torch.cuda.is_available()

    #image recuperation
    #-------------------------------------------------------------------------

    imgs=[]
    racine = '/home/guillaume/Documents/Env_DMLA/patients/'
    img_path = racine+patient

    # for subdir, dirs, files in os.walk(racine+patient):
    #     for file in files:
    #         imgs.append(racine+patient+ file)

    os.chdir(img_path)
    imgs = filter(os.path.isfile, os.listdir(img_path))
    imgs = [os.path.join(img_path, f) for f in imgs] # add path to each file
    imgs = natsort.natsorted(imgs,reverse=False)

    print("nombres d'images : ",len(imgs))


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
            print('Finetuning with patch_size : ',patch_size)
            print(100*('-'))

            parameters = {
            "patch_size": patch_size,
            "epoch_nb": epoch_nb,
            "batch_size": batch_size,
            "learning_rate":learning_rate,
            "weighted":weighted,  
            "sigma":sigma,         
            "shuffle":shuffle,
            "sampleTrue":sampleTrue,
            "threshold":threshold}

            if weighted:
                path_results = '/home/guillaume/Documents/Env_DMLA/results/'+patient+'patch_size_'+str(patch_size)+'/weighted/sigma'+str(sigma)+'/'
            else:
                path_results = '/home/guillaume/Documents/Env_DMLA/results/'+patient+'patch_size_'+str(patch_size)+'/normal/'

            path_models = path_results+'models/'
            path_models_finetuned = path_results+'models_finetuned/'
            path_diff = path_results+'differences/'
            path_bcm = path_results+'BCM/'
            path_loss = path_results+'Loss/'
            path_reconstructed = path_results+'Reconstructed/'

            paths = {
            "patient" : patient,
            "path_results" : path_results,
            "path_models_finetuned" : path_models_finetuned,
            "path_models" : path_models,
            "path_diff" : path_diff,
            "path_bcm" : path_bcm,
            "path_loss" : path_loss,
            "path_reconstructed" : path_reconstructed}

            #create dirs
            for p in [path_models_finetuned,path_diff,path_bcm,path_loss,path_reconstructed]:
                if not os.path.exists(p):
                    os.makedirs(p)
            
            # for loss and diff comparaison
            diffs = []
            loss_arrays = []
            diffs_otsu = []
            loss_arrays_otsu = []

            #extend the images
            extend_imgs = extend_img(cropped_imgs,patch_size)

            #-------------------------------------------------------------------------
            #finetuning

            t1 = time.clock()


            for i in tqdm(range(len(cropped_imgs)-1)):
                
                print('Processing : ',i,i+2)
                
                date1=(imgs[i].replace(racine+patient,''))  
                date2=(imgs[i+1].replace(racine+patient,''))  
                date1=(re.search(r'[2][0]\d{6}', date1)).group(0)
                date2=(re.search(r'[2][0]\d{6}', date2)).group(0)
                print(date1,'to',date2)
                pairs_img = cropped_imgs[i:i+2]
                pairs_extended = extend_imgs[i:i+2]
                
            #     ---------------------------------------- Loader part ----------------------------------------------

                loader = build_loader(pairs_img,pairs_extended,patch_size,batch_size,shuffle) #training dataset
                
            #     ----------------------------------- Pretrain loading part ------------------------------------------

                for subdir, dirs, files in os.walk(path_models):
                    for file in files:
                        if 'pkl' in file:
                            # if date1+'_to_'+date2 in file:    con pretrain sur toute la série donc il n'y a qu'un seul modèle de pretraining
                            model = path_models+file
                            print('pretrained : ',model)
                
                encoder12, decoder12 = torch.load(model)
                encoder21, decoder21 = torch.load(model)
                
                if gpu:
                    print('GPU')
                    encoder12 = encoder12.cuda()  # On GPU
                    decoder12 = decoder12.cuda()  # On GPU
                    encoder21 = encoder21.cuda()  # On GPU
                    decoder21 = decoder21.cuda()  # On GPU
                

            #     -------------------------------------- Finetuning part ----------------------------------------------


                epoch_loss_list, epoch_loss12_list, epoch_loss21_list, run_names = train(loader,encoder12,decoder12,encoder21,decoder21,date1,date2,parameters,paths)
                
                del loader
                del encoder12
                del decoder12
                del encoder21
                del decoder21
                
                #     ------------------------------------ Reconstruction part ---------------------------------------------

                # We create a datasetloader to encode the images
                
                bands_nb = 1
                flat = pairs_img[0].flatten()
                index_not_nan = np.where(np.isnan(flat) == False)
                index_not_nan = index_not_nan[0]

                image_enc = ImageDataset(pairs_extended[0], pairs_extended[1], patch_size,list(index_not_nan))  # we create a dataset with tensor patches
                batch_size = 1000
                loader = dsloader(image_enc, batch_size, shuffle=False)
                # avec drop last false
                criterion = nn.MSELoss(reduce=False)

                # We encode-decode the images and calculate the reconstruction error for the model obtained after every epoch
                
                # on prend juste la derniere epoque du finetuning

                #     for best_epoch in range(1):

                best_epoch = epoch_nb-1
                # We load the model
                best_epoch_loss12 = epoch_loss12_list[best_epoch]
                best_epoch_loss21 = epoch_loss21_list[best_epoch]
                best_encoder12, best_decoder12 = torch.load(path_models_finetuned+'id_'+date1+'_to_'+date2+'_ae12-model_ep_'+str(best_epoch)+"_loss_"+str(round(best_epoch_loss12, 7))+"_run_"+run_names[best_epoch]+'.pkl')
                best_encoder21, best_decoder21 = torch.load(path_models_finetuned+'id_'+date1+'_to_'+date2+'_ae21-model_ep_'+str(best_epoch)+"_loss_"+str(round(best_epoch_loss21, 7))+"_run_"+run_names[best_epoch]+'.pkl')


                if gpu:
                    best_encoder12 = best_encoder12.cuda()  # On GPU
                    best_encoder21 = best_encoder21.cuda()  # On GPU
                    best_decoder12 = best_decoder12.cuda()  # On GPU
                    best_decoder21 = best_decoder21.cuda()  # On GPU

                # We create empty arrays for reconstruction error images and for the reconstructed images
                    
                new_coordinates_reconstructed12 = []
                new_coordinates_loss_mean12 = []
                new_coordinates_reconstructed21 = []
                new_coordinates_loss_mean21 = []

                #We switch to the evaluation mode
                best_encoder12.eval()
                best_decoder12.eval()
                best_encoder21.eval()
                best_decoder21.eval()

                for batch_idx, (data1, data2, _) in enumerate(loader):  # we load batches from model
                    if gpu:
                        data1 = data1.cuda(async=True)
                        data2 = data2.cuda(async=True)
                    encoded12 = best_encoder12(Variable(data1))
                    decoded12 = best_decoder12(encoded12)
                    encoded21 = best_encoder21(Variable(data2))
                    decoded21 = best_decoder21(encoded21)

                    # we calculate batch loss to optimize the model

                    #             if weighted
                    if weighted:
            #                 creation des filtres gaussiens
                        weight = torch.from_numpy(gkern2(patch_size, sigma)).float().expand(decoded12.shape[0], 1, patch_size, patch_size)
                        if gpu:
                            weight = weight.cuda()

                        loss12 = criterion(decoded12*Variable(weight), Variable(data2)*Variable(weight))
                        loss21 = criterion(decoded21*Variable(weight), Variable(data1)*Variable(weight))

                    else:
                        loss12 = criterion(decoded12, Variable(data2))
                        loss21 = criterion(decoded21, Variable(data1))


                    #We transform the loss values in the array
                    loss_mean12 = loss12.view(-1, bands_nb, patch_size*patch_size).mean(2).mean(1)
                    loss_mean21 = loss21.view(-1, bands_nb, patch_size*patch_size).mean(2).mean(1)

                    if gpu:
                        new_coordinates_loss_batch_mean12 = loss_mean12.data.cpu().numpy()
                        new_coordinates_batch12 = decoded12.data.cpu().numpy()
                        new_coordinates_loss_batch_mean21 = loss_mean21.data.cpu().numpy()
                        new_coordinates_batch21 = decoded21.data.cpu().numpy()
                    else:
                        new_coordinates_loss_batch_mean12 = loss_mean12.data.numpy()
                        new_coordinates_batch12 = decoded12.data.numpy()
                        new_coordinates_loss_batch_mean21 = loss_mean21.data.numpy()
                        new_coordinates_batch21 = decoded21.data.numpy()

                        
                    new_coordinates_loss_mean12.append((new_coordinates_loss_batch_mean12))
            #         on prend le milieu de chaque patch du batch
                    new_coordinates_reconstructed12.append([new_coordinates_batch12[i][0][int(patch_size/2), int(patch_size/2)] for i in range(new_coordinates_batch12.shape[0])])

                    new_coordinates_loss_mean21.append((new_coordinates_loss_batch_mean21))
                    new_coordinates_reconstructed21.append([new_coordinates_batch21[i][0][int(patch_size/2), int(patch_size/2)] for i in range(new_coordinates_batch21.shape[0])])
                    
                    if (batch_idx + 1) % 200 == 0:
                        print('Encoding : [{}/{} ({:.0f}%)]'.format(
                            (batch_idx + 1) * batch_size, len(loader.dataset),
                            100. * (batch_idx + 1) / len(loader)))

                new_coordinates_loss_mean12 = np.concatenate(new_coordinates_loss_mean12)
                new_coordinates_loss_mean21 = np.concatenate(new_coordinates_loss_mean21)
                new_coordinates_reconstructed12 = np.concatenate(new_coordinates_reconstructed12)
                new_coordinates_reconstructed21 = np.concatenate(new_coordinates_reconstructed21)
                
                

                #     ------------------------------------ Assembling part ---------------------------------------------
                

                H,W = pairs_img[0].shape
                image_array_loss1 = np.zeros(H*W)*np.NaN
                image_array_loss2 = np.zeros(H*W)*np.NaN
                image_array_reconstructed1 = np.zeros(H*W)*np.NaN
                image_array_reconstructed2 = np.zeros(H*W)*np.NaN

                for i in range(index_not_nan.shape[0]):

                    image_array_loss1[index_not_nan[i]]=new_coordinates_loss_mean12[i]
                    image_array_loss2[index_not_nan[i]]=new_coordinates_loss_mean21[i]
                    image_array_reconstructed1[index_not_nan[i]]=new_coordinates_reconstructed12[i]
                    image_array_reconstructed2[index_not_nan[i]]=new_coordinates_reconstructed21[i]

                
                # We create a loss image in new coordinate system for reconstruction of 2nd image from the 1st
                image_array_loss1 = np.reshape(image_array_loss1, (H, W))
                # cv2.imwrite(path_loss+"Loss12_"+date1+'_to_'+date2+'_epoch_nb_'+str(best_epoch+1)+'.png',normalize(image_array_loss1)*255)

                # We create a loss image in new coordinate system of 1st image from the 2nd
                image_array_loss2 = np.reshape(image_array_loss2, (H, W))
                # cv2.imwrite(path_loss+"Loss21_"+date1+'_to_'+date2+'_epoch_nb_'+str(best_epoch+1)+'.png',normalize(image_array_loss2)*255)
                
                # We reconstruct the 2nd image from the 1st
                image_array_reconstructed1 = np.reshape(image_array_reconstructed1, (H, W))
                cv2.imwrite(path_reconstructed+"Reconstructed12_"+date1+'_to_'+date2+'_epoch_nb_'+str(best_epoch+1)+'.png',normalize(image_array_reconstructed1)*255)
                
                # We reconstruct the 1st image from the 2nd
                image_array_reconstructed2 = np.reshape(image_array_reconstructed2, (H, W))
                cv2.imwrite(path_reconstructed+"Reconstructed21_"+date1+'_to_'+date2+'_epoch_nb_'+str(best_epoch+1)+'.png',normalize(image_array_reconstructed2)*255)

                # We calculate the average reconstruction error image
                image_array_loss = np.divide((image_array_loss1+image_array_loss2), 2)
                #normalize loss for visual
                image_array_loss = normalize(image_array_loss)*255
                cv2.imwrite(path_loss+"Loss_"+date1+'_to_'+date2+'_epoch_nb_'+str(best_epoch+1)+'.png',image_array_loss)
                #Here we calculate change map using otsu threshold on the average reconstruction error image


                result = otsu(image_array_loss1,image_array_loss2, H, W,date1,date2,parameters,paths,loss_arrays,loss_arrays_otsu)

                #diff en vert
                img2 = pairs_img[1]
                img2color = np.stack((img2,img2, img2), axis=2)
                mask = np.where(result==0,1,0)
                img2color[:,:,0]=img2color[:,:,0]*mask
                img2color[:,:,2]=img2color[:,:,2]*mask

                #img1 en RV et img2 en B
                composition = np.stack((pairs_img[0],pairs_img[0], pairs_img[1]), axis=2)

                #difference avec otsu thresholding
                diff = normalize(abs(pairs_img[1]-pairs_img[0]))*255
                diffs.append(diff)
                
                threshold_global_otsu = filters.threshold_otsu(diff[~np.isnan(diff)].flatten())
                global_otsu = diff >= threshold_global_otsu
                diffs_otsu.append([global_otsu,int(threshold_global_otsu)])

                f, axarr = plt.subplots(2,3,figsize=(30,20))

                #     plt.tight_layout()
                plt.rc('font', size=20)    

                axarr[0][0].imshow(pairs_img[0],cmap='gray')
                axarr[0][1].imshow(pairs_img[1],cmap='gray')
                axarr[0][2].imshow(global_otsu, cmap='gray')
                axarr[1][0].imshow(composition)
                axarr[1][1].imshow(img2color)
                axarr[1][2].imshow(result,cmap='gray')

                axarr[0][0].set_title(date1)
                axarr[0][1].set_title(date2)
                axarr[0][2].set_title('Diff with otsu ='+str(int(threshold_global_otsu)))
                axarr[1][0].set_title('Composition')
                axarr[1][1].set_title('Detected changes')
                axarr[1][2].set_title('Change Map Algorithm')
                plt.suptitle("Patch size = "+str(patch_size)+" | Threshold = "+str(threshold))

                f.savefig(path_diff+"Diff_"+date1+'_to_'+date2+'_threshold_'+str(threshold)+'.png')


            t2 = time.clock()
            total_time_learning = t2 - t1
            total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
            print("Total time learning =", total_time_learning)

            if make_score:
                try:
                    print('score succeeded')
                    score(path_original,path_true,path_bcm,path_results+'score/')
                except:
                    print('score failed')
            
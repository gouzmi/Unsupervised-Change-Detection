import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys, re
import math
from skimage import measure
import time, datetime
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from matplotlib.ticker import MaxNLocator
import natsort 

def bouding_box(path_original,to_transform):
    
    original = cv2.imread(path_original,0)
    mask = np.where(original==0,original,np.NaN)
    
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

    to_transform = to_transform[y_min:y_max,x_min:x_max]
    original = original[y_min:y_max,x_min:x_max]
    
    to_transform = np.where(original==0,np.NaN, to_transform)

    return to_transform

def mask_bouding_box(path_original):
    
    original = cv2.imread(path_original,0)
    mask = np.where(original==0,original,np.NaN)
    
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

    original = original[y_min:y_max,x_min:x_max]
    
    original = np.where(original==0,np.NaN, 1)

    return original

def difference(path_true,path_original=None):
    
#     fonction qui fait les différences entre les segmentations des lésions
    os.chdir(path_true)
    files = filter(os.path.isfile, os.listdir(path_true))
    files = [os.path.join(path_true, f) for f in files] # add path to each file
    files = natsort.natsorted(files,reverse=False)
    
    diffs = []
    for i in range(len(files)-1):
        diff = np.maximum(0,normalize(cv2.imread(files[i+1],0))-normalize(cv2.imread(files[i],0)))
        diffs.append(diff)


    if path_original != None:
        for i in range(len(diffs)):
            diffs[i] = bouding_box(path_original,diffs[i])
    
    return diffs

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


def score(path_original,path_true,path_to_score,path_store):
    
#     make path if not exists
    if not os.path.exists(path_store):
        os.makedirs(path_store)

    # get one image from original folder to make bouding box
    for subdir, dirs, files in os.walk(path_original):
        for file in files:
            path_original = path_original + file
            print('original image : ',path_original)
            break
    
#     get imgs to score ordered by time
    os.chdir(path_to_score)
    files = filter(os.path.isfile, os.listdir(path_to_score))
    files = [os.path.join(path_to_score, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    to_score = []
    for i in range(len(files)):
        to_score.append([mask_bouding_box(path_original)*normalize(cv2.imread(files[i],0)),(files)[i][-54:-34]])
        
#     get difference of segmentations
    diffs = difference(path_true,path_original)
    all_score = []

#     plot and save : Net | Seg difference | Confusion matrix & recall,precision,f1, accuracy
    for i in range(len(to_score)):

        f = plt.figure(figsize=(30,10))
        plt.rc('font', size=20) 

        y_pred = to_score[i][0].flatten()
        y_true = diffs[i].flatten()
        y_pred = y_pred[~np.isnan(y_pred)]
        y_true = y_true[~np.isnan(y_true)]
        r,p,f1,acc = metrics.recall_score(y_true, y_pred), metrics.precision_score(y_true, y_pred), metrics.f1_score(y_true, y_pred), metrics.accuracy_score(y_true, y_pred)
        print('avec image blanche ','recall ',metrics.recall_score(y_true, (y_pred*0)+1),'precision ', metrics.precision_score(y_true, (y_pred*0)+1),'f1 ', metrics.f1_score(y_true, (y_pred*0)+1),'accuracy ', metrics.accuracy_score(y_true, (y_pred*0)+1))
        all_score.append([r,p,f1,acc])
        cm = metrics.confusion_matrix(y_true,y_pred)

        ax1=f.add_subplot(1, 3, 1) # 1 lignes 2 colonnes
        ax1.imshow(to_score[i][0],cmap='gray')
        ax2=f.add_subplot(1, 3, 2) # 1 lignes 2 colonnes
        ax2.imshow(diffs[i],cmap='gray')
        ax3=f.add_subplot(1, 3, 3) # 1 lignes 2 colonnes
        ax3.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.set(ylabel='True label',xlabel='Predicted label')

        for (j,ii),label in np.ndenumerate(cm):
                ax3.text(ii,j,label,ha='center',va='center',bbox=dict(facecolor='white', alpha=0.5))

        ax1.set_title("Net")
        ax2.set_title("Segmentation")
        ax3.set_title("Confusion Matrix")

        plt.suptitle((str(to_score[i][1])+" | Recall = % 5.2f , Precision = % 5.2f , f1_score = % 5.2f , Accuracy = % 5.2f" %(r,p,f1,acc)))
        f.savefig(path_store+"Score_"+str(to_score[i][1])+'.png')
        
    all_score = pd.DataFrame(all_score)
    all_score.columns = ['recall','precision','f1_score','accuracy']
    print(all_score.describe())
    all_score.describe().to_csv(path_store+'scores.csv',header=True)

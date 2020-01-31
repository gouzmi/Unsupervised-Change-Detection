import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, sys


def plotting(epochs, loss_array,patch_size,batch_size,learning_rate, path,run_name):

    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, epochs+1), loss_array)
    # ax.set_xticks(range(0, epochs+1, 50))
    ax.set(xlabel='epochs', ylabel='loss',
           title=('Loss epochs ='+ str(epochs)+' , Patch ='+str(patch_size)+' , Batch ='+str(batch_size)+' , Lrate ='+str(learning_rate)))
    ax.grid()
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(path+'_run_'+str(run_name)+' | nb_epoch'+str(epochs)+'.png')
    plt.close(fig)
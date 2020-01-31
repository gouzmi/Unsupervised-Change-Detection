from codes.score import score

#patient = '102_SE_J/IR OD/'
# patient ='005_AU_R/IR OD/'
patient ='001_AF_S/IR OD/spe/'
# patient = '003_AL_J/IR OD/'
# patient ='008_BA_D/IR OG/spe/'
# patient = '010_BA_J/IR OG/'

path_original = '/media/guillaume/OS/Users/gouzm/Documents/DMLA-TimeLapse-Align-corrected/001_AF_S/IR OD/aligned_00_20110727_mapped_DCV2.png'
path_true = '/media/guillaume/OS/Users/gouzm/Documents/Env_DMLA/results/référence pour score/DMLA-TimeLapse-Align-Segmented-manu/'+patient


patch_sizes = [7,9,13]
sigmas = [5,12]

for patch_size in patch_sizes:
    for sigma in sigmas:

        path_results = '/home/guillaume/Documents/Env_DMLA/results/'+patient+'patch_size_'+str(patch_size)+'/weighted/sigma'+str(sigma)+'/'
        path_bcm = path_results+'BCM/'
        try:
            score(path_original,path_true,path_bcm,path_results+'score/')
            print('score succeeded')
        except:
            print('score failed')

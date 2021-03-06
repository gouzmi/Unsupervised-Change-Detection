# Unsupervised-Change-Detection

Implementation of :
https://www.researchgate.net/publication/342534299_Analyzing_Age-Related_Macular_Degeneration_Progression_in_Patients_with_Geographic_Atrophy_Using_Joint_Autoencoders_for_Unsupervised_Change_Detection

Unsupervised Change Detection using Deep Learning Techniques for Age-Related Macular Degeneration Progression

This repo shows a novel approach to unsupervised change detection.
We used it on retinal images of AMD patients.
It would thus make it possible to describe its evolution in order to build a predictive model.

The folder codes contains all the functions we use for processing the images, building the loaders ect.

The folder models contains the autoencoder in Pytorch.

Our algorithm is in two parts : pretraining.py and finetuning.py . Don't forget to change the paths for your images.

To launch all the algorithm just run pipeline.py.

Example:

![Image1](example.png)


Here is the architecture of our neural networks :

![Image1](no3d.png)


Our algorithm  :

![Image2](schema.png)

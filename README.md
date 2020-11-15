# Aramco Upstream Solutions Technathon 2020

## Digital rock model reconstruction

> This repository contains results for each checkpoint of the following technathon

>Tags: 3D reconstruction, DigitalRock, Rock, Pore, Segmentation, Neural Network, Deep Learning, Deep, Learning, grains, SEM, QEMSCAN, Segmentation Neural Network, Tensorflow, Keras, CNN, Convolutional Neural Network, GAN, Generative adversarial network

### Implemented by: 
* Victoria Dochkina
* Vladislav Alekseev
* Egor Fisher
## Guideline
In **checkpoint_2** folder you can find the results of the first week of the technathon.

## Problem formulation

The use of digital rock models provides relatively inexpensive, non-destructive methodology for studying rocks. Digital rocks models are being used in the oil and gas industry to numerically simulate several properties such as absolute permeability, relative permeability, electrical and elastic properties. These simulations can be done in 2D and 3D. Unfortunately, to date, there is not deterministic relationship to convert 2D results to 3D. Thus, numerical simulation performed on 3D digital models is preferred.

To obtain 3D digital rock models, computational tomography is typically performed on a rock sample. The procedure is relatively slow. Other types of rock images, such as thin section images, are more readily available. These images are 2D images. The question is: can these 2D images be utilized to construct realistic 3D models to be used for accurate numerical simulation?

## Brief summary of the results:

### 1st week results
In **checkpoint_2** folder you can find the results of the first week of the technathon.
Binarization of provided raw CT images was recieved.

3d binarized reconstruction of Beta_1_0.2 dataset:

<img src="https://github.com/ddvika/Aramco_technathon_2020/blob/main/checkpoint_2/imgs/binary_reconstruction.png" width="1000" >

<center> Figure 1. Predicted by our algorithm vs Original B&W 3d reconstruction </center>

Example of porosity prediction for Beta_1_0.2 dataset:

<img src="https://github.com/ddvika/Aramco_technathon_2020/blob/main/checkpoint_2/imgs/porosity_prediction.png" width="600" >
<center> Figure 1. Predicted by our algorithm vs Original porosity </center>


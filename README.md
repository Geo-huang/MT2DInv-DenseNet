# MT2DInv-DenseNet
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14120745.svg)](https://doi.org/10.5281/zenodo.14120745)

The source codes and data corresponds to manuscript: N. Yu, C. Wang, H. Chen et al., A two-dimensional magnetotelluric deep
learning inversion approach based on improved Dense Convolutional Network. Computers and
Geosciences (2024), doi: https://doi.org/10.1016/j.cageo.2024.105765.

Followings are descriptions about the two uploaded folders:

#Folder 1：unchanged_DenseNet

This folder contains two Python programs that combine the standard DenseNet network with the magnetotelluric
2D inversion problem to solve the magnetotelluric 2D inversion problem. They are the training function file
‘train_MTinv_densenet.py’ and the prediction function file ‘predict_MTinv_densenet.py’.

#Folder 2：improved Densenet

This folder contains two Python programs, which improve the algorithm based on the standard DenseNet, solve
the two-dimensional magnetotelluric inversion problem and improve the inversion reliability. They are the
training function file ‘train_MTinv_iDenseNet.py’ and the prediction function file ‘predict_MTinv_iDenseNet.py’.

These programs are developed and trained using Python 3.9 and PyTorch 1.13 framework for deep learning model development and training.

# MT2DInv-DenseNet
A 2-D MT deep learning inversion code based on improved DenseNet
The main source code for synthetic data experiments， also the main source code for the article.:

#1. file： unchanged_DenseNet
This file contains two Python programs that combine the standard DenseNet network with the magnetotelluric
2D inversion problem to solve the magnetotelluric 2D inversion problem. They are the training function file
‘train_MTinv_densenet.py’ and the prediction function file ‘predict_MTinv_densenet.py’.

#2. file： improved Densenet
This file contains two Python programs, which improve the algorithm based on the standard DenseNet, solve
the two-dimensional magnetotelluric inversion problem and improve the inversion reliability. They are the
training function file ‘train_MTinv_iDenseNet.py’ and the prediction function file ‘predict_MTinv_iDenseNet.py’.

These programs are developed and trained using Python 3.9 and PyTorch 1.13 framework for deep learning model development and training.

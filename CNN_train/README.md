# Numpy-CNN
A numpy-only implementation of a Convolutional Neural Network, from the ground up.

*Written by Alejandro Escontrela for [this article](https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1) on CNNs.*

## Purpose

To gain a quality understanding of convolutional neural networks and what makes them peform so well, I constructed one from scratch with NumPy. This CNN is in no way intended to replace popular DL frameworks such as Tensorflow or Torch, *it is instead meant to serve as an instructional tool.*

## Training the network

To train the network on your machine, first install all necessary dependencies using:


`$ pip install -r requirements.txt`

Afterwards, you can train the network using the following command: 

`$ python3 train_cnn.py 'save.pkl'`

Or using:

`$ ./run.sh'`

After the CNN has finished training, a .pkl file containing the network's parameters is saved to the directory where the script was run.

The dataset I use is the same dataset from hw.  To train a different dataset (t10k image). using the following command:

`$ python3 train_cnn_2.py 'save.pkl'`

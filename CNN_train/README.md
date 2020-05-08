# Numpy-CNN
A numpy-only implementation of a Convolutional Neural Network, from the ground up.
This CNN doesn't perform as well as tensorflow, but it is constructed from scratch with Numpy.

*Edited from codes Alejandro Escontrela for [this article](https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1) on CNNs.*


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

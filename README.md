# CISC5800Final Neural Network from scratch

Program structure:

neuralnet.py:
    This is the module containing the implementation of the neural network. This module contains 2 classes Layer and Network.
    The network can be initialized using Network() constructor, and construct_network() will add layers to the network.

adult.py:
    Run this script to fully test out the neuralnet module. This script reads in the dataset under the filename "adult.data",
        pre-process, account for imbalanced data, and run it through a neural network with 1 hidden layer of 2 neurons.
    The script can be easily modified to test different training function by replacing momentum_train with train or adaptive_train


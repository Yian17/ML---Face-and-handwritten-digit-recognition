# Overview
- This projects includes two classification problem: Digit Recognition and Face Recognition

## Digit Recognition
<b>Data</b>
- Work with the MNIST dataset.
- The digits already seprated into a training and a test set.

<b>Network</b>
![Image of the Network](http://www.cs.toronto.edu/~guerzhoy/411/proj2/logreg.png)

<b>Cost Function</b>
The sum of the negative log-probabilities of all the training cases

## Face Recognition
Developed a single-hidden-layer neural network model with a custom loss function using PyTorch to classify faces extracted from 100,000 images
<b>Data</b>
- A subset of the FaceScrub dataset.
- The dataset consists of URLs of images with faces, as well as the bounding boxes of the faces.

<b>Network</b>
a single-hidden-layer fully-connected network

## Techniques
learning rate, batch normalization, dropout, and various optimizers，regularization

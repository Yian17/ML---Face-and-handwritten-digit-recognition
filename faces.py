from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random as random
import random as rd
import torch
from torch.autograd import Variable
from hashlib import sha256
import os
from scipy.io import loadmat
import torch.utils.data as Data

M = loadmat("mnist_all.mat")


def rm_bad_img():
    # Only found one bad image after doing SHA-256
    img = "drescher69.jpg"
    if os.path.isfile("cropped/" + img):
        os.remove("cropped/" + img)


#=====================Part 8=======================

def sub_x(batch_size, x, dim_h, t):
    idx = (t*batch_size)%dim_h
    x = x[idx:idx + batch_size]
    return x


def sub_y(batch_size, y_classes, dim_h, t):
    idx = (t*batch_size)%dim_h
    y_classes = y_classes[idx:idx + batch_size]
    return y_classes


def get_set(actors, train_size):
    training_sets = {}
    validation_sets = {}
    test_sets = {}

    for actor in actors:
        a = actor.split()[1].lower()
        temp = []
        for image in os.listdir("cropped_rgb/"):
            if a in image:
                # print "shape==========", a.shape
                temp.append(image)
        np.random.seed(7)
        np.random.shuffle(temp)
        training_sets[actor] = temp[25:25 + train_size]
        validation_sets[actor] = temp[20:25]
        test_sets[actor] = temp[0:20]
    return training_sets, validation_sets, test_sets


def get_several_y(set_size, act_lst):
    length = len(act_lst) # 6
    total_len = set_size * length #60
    y = np.zeros((total_len, length))
    count = 0
    position = 0
    for i in range(total_len):
        flag = 0
        for j in range(length):
            if j % length == position and flag == 0:
                y[i][j] = 1
                count += 1
                flag = 1
                if count == set_size:
                    position += 1
                    # print position
                    count = 0
    return y


def get_img_vector(input_set):
    l = np.zeros([len(input_set), 3072])

    i = 0
    for f_name in os.listdir("cropped_rgb/"):
        if f_name in input_set:

            path = "cropped_rgb/" + f_name
            im = imread(path)
            if im.shape == (32, 32):
                print "gray", f_name
                im = np.stack((im,)*3, -1)
            if im.shape == (32, 32, 4):
                # im = np.stack((im, )*0.75, -1)
                # train_y = np.delete(train_y, i, 0)
                im = np.delete(im, -1, 2)
                print "4", f_name
                print "shape!!", im.shape
                # continue

            im = im.flatten()
            l[i] = reshape(np.array(im), [1, 3072])
            # print "l[i]", l[i]
            i += 1
    l /= 255
    return l


def vec_set(training, validation, test):
    tr_vec = get_img_vector(training)
    # print "tr_vec: ", tr_vec
    va_vec = get_img_vector(validation)
    te_vec = get_img_vector(test)

    return tr_vec, va_vec, te_vec


def performance(set_x, set_y, model, dtype_float):
    train_x = Variable(torch.from_numpy(set_x), requires_grad=False).type(dtype_float)
    y_pred = model(train_x).data.numpy()
    acc = (np.mean(np.argmax(y_pred, 1) == np.argmax(set_y, 1))) * 100
    return acc

from faces import *

def part89():
    torch.manual_seed(10)
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    train_size = 75
    train_set, val_set, test_set = get_set(act, train_size)
    train_x = np.empty([0, 3072])
    valid_x = np.empty([0, 3072])
    test_x = np.empty([0, 3072])

    # Delete training size for Gilpin
    actual_length = len(train_set[act[1]])
    expect_length = 60
    delete_length = actual_length - expect_length

    for i in range(len(act)):
        act_train, act_val, act_test = vec_set(train_set[act[i]], val_set[act[i]], test_set[act[i]])
        if i == 2:
            for j in range(delete_length):
                act_train = np.delete(act_train, -1, 0)
        train_x = np.vstack((train_x, act_train))
        valid_x = np.vstack((valid_x, act_val))
        test_x = np.vstack((test_x, act_test))
        print "i, act", i, act[i]

    train_y = get_several_y(train_size, act)
    valid_y = get_several_y(5, act)
    test_y = get_several_y(20, act)
    # print "========train_x,y========"
    # print train_x, train_x.shape
    # print train_y, train_y.shape
    # print "========test_x========="
    # print test_x
    # print test_y


    print actual_length, expect_length
    # for j in range(actual_length - expect_length):

    for i in range(train_size - expect_length):
        train_y = np.delete(train_y, train_size + 1, 0)
    print "================train_y shape============"
    print train_y.shape
    # print train_y[train_size - 1:train_size+70]
    print "================train_x shape============"
    print train_x.shape

    dim_x = 32 * 32 * 3
    dim_h = 300
    dim_out = 6

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_idx = np.random.permutation(range(train_x.shape[0]))[:1000]
    x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
    # print "train idx", train_idx, train_idx.shape
    # print "train x", train_x.shape

    y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)

    # Define the neural network model
    model = torch.nn.Sequential(
            torch.nn.Linear(dim_x, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    batch_size = 20

    # Training the model
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    itr_idx, train_perform, valid_perform, test_perform = [], [], [], []

    for t in range(1500):
        xt = sub_x(batch_size, x, dim_h, t)
        y_class = sub_y(batch_size, y_classes, dim_h, t)
        y_pred = model(xt)
        loss = loss_fn(y_pred, y_class)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0 or t == 1499:
            itr_idx.append(t)
            print "t:", t
            train_acc = performance(train_x, train_y, model, dtype_float)
            train_perform.append(train_acc)
            print "training set accuracy: " + str(train_acc)

            test_acc = performance(test_x, test_y, model, dtype_float)
            test_perform.append(test_acc)
            print "testing set accuracy:  " + str(test_acc)

            valid_acc = performance(valid_x, valid_y, model, dtype_float)
            valid_perform.append(valid_acc)
            print("validation set accuracy:  " + str(valid_acc) + "%\n")

    plt.figure()
    plt.plot(itr_idx, train_perform, color='blue', marker='o', label="Training")
    plt.plot(itr_idx, test_perform, color='green', marker='o', label="Test")
    plt.plot(itr_idx, valid_perform, color='black', marker='o', label="valid")

    plt.legend(loc="best")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Performance on Accuracy")
    savefig('part8')
    plt.show()

    plt.figure()
    plt.axis('off')

    print "======part9========"


    weight = model[0].weight.data.numpy()

    h = []
    acts = ['Lorraine Bracco', 'Angie Harmon']
    for i in [0, 2]:
        hid = model[2].weight.data.numpy()[i,:].argsort()[-4:][::-1]
        h.append(hid)

    for j in range(len(h)):
        idx = 0
        for k in range(len(h[j])):
            print "picture for", acts[j]
            W = weight[h[j][k], :].reshape((32, 32, 3))
            W = (W[:,:,0] + W[:,:,1] + W[:,:,2])/255
            plt.axis('on')
            plt.imshow(W, cmap="coolwarm")
            idx += 1
            savefig("part9" + acts[j] + str(idx))
            plt.show()


# part89()


################################################################################
# ===================== PART 10 ================================================

import os
import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from numpy import float32
import numpy.random as rd

import torch.nn as nn

#os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')

# a list of class names
from caffe_classes import class_names


# We modify the torchvision implementation so that the features
# after the final pooling layer is easily accessible by calling
#       net.features(...)
# If you would like to use other layer features, you will need to
# make similar modifications.

def performance(set_x, set_y, model, dtype_float):
    train_x = Variable(torch.from_numpy(set_x), requires_grad=False).type(dtype_float)
    y_pred = model(train_x).data.numpy()
    acc = (np.mean(np.argmax(y_pred, 1) == np.argmax(set_y, 1))) * 100
    return acc


class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)

        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x


def get_set(actors, train_size):
    training_sets = {}
    validation_sets = {}
    test_sets = {}

    for actor in actors:
        a = actor.split()[1].lower()
        temp = []
        for image in os.listdir("cropped_rgb227_1/"):
            if a in image:
                temp.append(image)
        for image in os.listdir("cropped_rgb227_2/"):
            if a in image:
                temp.append(image)
        np.random.seed(7)
        np.random.shuffle(temp)
        training_sets[actor] = temp[25:25 + train_size]
        validation_sets[actor] = temp[20:25]
        test_sets[actor] = temp[0:20]
    return training_sets, validation_sets, test_sets

# def get_set():


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


def part10():

    model = MyAlexNet()
    model.eval()

    torch.manual_seed(0)
    dim_x = 256 * 6 * 6
    dim_h = 300
    dim_out = 6

    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    train_size = 60
    train_set, valid_set, test_set = get_set(act, train_size)
    train_total, valid_total, test_total= [], [], []

    # Delete training size for Gilpin
    actual_length = len(train_set[act[1]])
    expect_length = 60
    delete_length = actual_length - expect_length
    for i in range(delete_length):
        to_remove = train_set[act[1]][-1]
        train_set[act[1]].remove(to_remove)

    for i in range(len(act)):
        train_x = train_set[act[i]]
        train_total.append(train_x)
        test_x = test_set[act[i]]
        test_total.append(test_x)
        valid_x = valid_set[act[i]]
        valid_total.append(valid_x)

    x_train, x_test, x_valid = [], [], []

    for row in train_total:
        for col in row:
            x_train.append(col)

    for row in test_total:
        for col in row:
            x_test.append(col)

    for row in valid_total:
        for col in row:
            x_valid.append(col)

    train_x, test_x, valid_x = [], [], []

    for filename in x_train:
        # Read an image
        try:
            im = imread("./cropped_rgb227_1/" + filename)
        except Exception:
            im = imread("./cropped_rgb227_2/" + filename)

        if im.shape == (227, 227):
            im = np.stack((im,)*3, -1)
        if im.shape == (227, 227, 4):
            im = np.delete(im, -1, 2)
        im = im[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32)

        imv = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)

        train_x += [model.forward(imv).data.numpy()]

    train_x = np.array(train_x)
    train_x = np.reshape(train_x, (np.shape(train_x)[0], dim_x))
    train_y = get_several_y(train_size, act)

    # delete y for Gilpin
    for i in range(train_size - expect_length):
        train_y = np.delete(train_y, train_size + 1, 0)

    for filename in x_test:
        # Read an image
        try:
            im = imread("./cropped_rgb227_1/" + filename)
        except Exception:
            im = imread("./cropped_rgb227_2/" + filename)
        if im.shape == (227, 227):
            im = np.stack((im,)*3, -1)
        if im.shape == (227, 227, 4):
            im = np.delete(im, -1, 2)
        im = im[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32)

        imv = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)

        test_x += [model.forward(imv).data.numpy()]

    test_x = np.array(test_x)
    test_x = np.reshape(test_x, (np.shape(test_x)[0], dim_x))
    test_y = get_several_y(20, act)

    for filename in x_valid:
        # Read an image
        try:
            im = imread("./cropped_rgb227_1/" + filename)
        except Exception:
            im = imread("./cropped_rgb227_2/" + filename)
        if im.shape == (227, 227):
            im = np.stack((im,)*3, -1)
        if im.shape == (227, 227, 4):
            im = np.delete(im, -1, 2)
        im = im[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32)

        imv = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)

        valid_x += [model.forward(imv).data.numpy()]

    valid_x = np.array(valid_x)
    valid_x = np.reshape(valid_x, (np.shape(valid_x)[0], dim_x))
    valid_y = get_several_y(5, act)


    learning_rate = 1e-4

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_result, test_result, valid_result = [], [], []
    count = []

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()     #set loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #set learning rate

    x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)

    for t in range(200):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 15 == 0:
            print "t:", t
            count.append(t)
            train_perform = performance(train_x, train_y, model, dtype_float)
            print "train_perform", train_perform
            train_result.append(train_perform)

            test_perform = performance(test_x, test_y, model, dtype_float)
            print "test_perform", test_perform
            test_result.append(test_perform)

            valid_perform = performance(valid_x, valid_y, model, dtype_float)
            print "valid_perform", valid_perform
            valid_result.append(valid_perform)

    plt.plot(count, train_result, label='Training', color='blue', marker='o',)
    plt.plot(count, test_result, label='Test', color='green', marker='o',)
    plt.plot(count, valid_result, label='Validation', color='black', marker='o',)
    plt.title('Learning Curve')
    plt.xlabel('Number of t')
    plt.ylabel('Performance')
    plt.legend()
    plt.savefig('part10.jpg')
    plt.show()
    plt.close()

# part10()

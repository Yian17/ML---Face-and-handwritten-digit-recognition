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


import cPickle


import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

# =========================== PART 1 ==========================================
def part1():

    f, axarr = plt.subplots(10, 10, sharex='all', sharey='all')
    rd.seed(1)
    for ax1 in range(10):
        lst = rd.sample(M["train"+str(ax1)], 10)
        for ax2 in range(10):
            axarr[ax1, ax2].imshow(lst[ax2].reshape((28,28)), cmap=cm.gray)
            axarr[ax1, ax2].axis('off')

    plt.show()
    plt.savefig("part1.jpg")

# part1()

# =========================== PART 2 ==========================================
def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y), 0), (len(y), 1))

def part2(x, W, b):
    # I recommend you divide the data by 255.0
    #x /= 255.
    # x: (784, ) -> (784, 1)
    if x.shape == (784, ):
        x = x[:, np.newaxis]
    # initialize Weight and bias matrix to all zero
    # W = np.zeros((784, 10))
    # b = np.zeros((10, 1))
    return softmax(np.dot(W.T, x) + b)

# =========================== PART 3 ==========================================

def cf(x, y, W, b):
    p = part2(x, W, b)
    return -sum(y * log(p))

# computes the gradient of the cost function with respect to the weights and biases of the network
def df(x, y, W, b):

    p = part2(x, W, b)
    partial_Co = p - y  # (10, 1)

    if x.shape == (784, ):
        x = x[:, np.newaxis]

    # x -> (784, 1) partial_Co -> (1, 10)
    partial_CW = dot(x, partial_Co.T) #reshape((1, 10))
    partial_Cb = dot(partial_Co, ones((partial_Co.shape[1], 1)))

    return partial_CW, partial_Cb

def cost_part3(x, y, W, b):
    if x.shape == (784, ):
        x = x[:, np.newaxis]
    p = softmax(np.dot(W.T, x) + b)
    cost_val = -sum(y * log(p))
    return cost_val

def part3():
    # I recommend you divide the data by 255.0
    x = M["test0"][10].T/255.  # x -> (784, )
    y = np.zeros((10, 1))  # true value of output
    y[0] = 1

    W = np.random.randn(784, 10) * 10e-5
    b = np.zeros((10, 1))
    par_CW, par_Cb = df(x, y, W, b)

    np.random.seed(5)
    for i in range(5):
        m = np.random.randint(0, 784)
        n = np.random.randint(0, 10)
        h = 0.00000001

        # respect to Weights
        weight_h = np.zeros((784, 10))
        weight_h[m][n] = h
        fd_weight = (cost_part3(x, y, W + weight_h, b) - cost_part3(x, y, W, b)) / h

        # respect to Bias
        bias_h = np.zeros((10, 1))
        bias_h[n][0] = h
        fd_bias = (cost_part3(x, y, W, b + bias_h) - cost_part3(x, y, W, b)) / h

        print "================================================================"
        print "at coordinates:(" + str(m) + ", " + str(n) + " )"
        print "finite difference with respect to weight: " + str(fd_weight)
        print "gradient of cost function with respect to the weights: " + str(par_CW[m][n])
        print "finite difference with respect to bias: " + str(fd_bias)
        print "gradient of cost function with respect to the bias: " + str(par_Cb[n][0])

# part3()

# =========================== PART 4 ==========================================

def gradient_descent(x, y, W, b, alpha, itr):
    EPS = 1e-5
    prev_w = W - 10 * EPS
    W_cp = W.copy()
    b_cp = b.copy()
    i = 0
    results = []
    while norm(W_cp - prev_w) > EPS and i < itr:
        i += 1
        prev_w = W_cp.copy()
        partial_CW, partial_Cb = df(x, y, W_cp, b_cp)
        W_cp -= alpha * partial_CW
        b_cp -= alpha * partial_Cb

        if i % 100 == 0:
            print "Iteration: ", i
            curr_W, curr_b = W_cp.copy(), b_cp.copy()
            results.append([i, curr_W, curr_b])

    return results, W_cp, b_cp

def part4():
    W = np.random.randn(784, 10) * 10e-5
    b = np.zeros((10, 1))

    training = np.empty((784, 0))
    tests = np.empty((784, 0))
    y_training = np.empty((10, 0))
    y_test = np.empty((10, 0))

    for i in range(0, 10):
        training = np.hstack((training, M["train"+str(i)].T/255.0))
        tests = np.hstack((tests, M["test"+str(i)].T/255.0))
        training_size = len(M["train"+str(i)])
        test_size = len(M["test"+str(i)])
        # one hot vector
        o_vector = np.zeros((10, 1))
        o_vector[i] = 1
        y_training = np.hstack((y_training, tile(o_vector, (1, training_size))))
        y_test = np.hstack((y_test, tile(o_vector, (1, test_size))))

    # calculating performance
    alpha = 0.00001
    plot_data, end_W, end_b = gradient_descent(training, y_training, W, b, alpha, 3000)

    train_accuracy, test_accuracy, itr_idx = [], [], []
    for i in plot_data:
        itr_idx.append(i[0])
        curr_w, curr_b = i[1], i[2]
        training_size, test_size = training.shape[1], tests.shape[1]

        train_correct, test_correct = 0, 0
        train_x = part2(training, curr_w, curr_b)
        test_x = part2(tests, curr_w, curr_b)

        for j in range(training_size):
            if y_training[:, j].argmax() == train_x[:, j].argmax():
                train_correct += 1
        for k in range(test_size):
            if y_test[:, k].argmax() == test_x[:, k].argmax():
                test_correct += 1

        train_accuracy.append(float(train_correct)/float(training_size))
        test_accuracy.append(float(test_correct)/float(test_size))


    plt.figure()
    plt.plot(itr_idx, train_accuracy, color='blue', marker='o', label="Training")
    plt.plot(itr_idx, test_accuracy, color='green', marker='o', label="Test")
    plt.legend(loc="best")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Performance on Accuracy")
    savefig('part4_learning_curves')

    plt.figure()
    plt.axis('off')
    for i in range(0, 10):
        plt.subplot(1, 10, i + 1).axis('off')
        number = end_W[:, i].reshape((28, 28))
        plt.imshow(number, cmap="coolwarm")
    savefig("part4_weights.png")

# part4()

# =========================== PART 5 ==========================================

def grad_descent_m(x, y, W, b, alpha, itr, momentum):
    EPS = 1e-5
    prev_w = W - 10 * EPS
    W_cp = W.copy()
    b_cp = b.copy()
    i = 0
    results = []
    while norm(W_cp - prev_w) > EPS and i < itr:
        i += 1
        prev_w = W_cp.copy()
        partial_CW, partial_Cb = df(x, y, W_cp, b_cp)

        W2, b2 = W.copy(), b.copy()
        W_cp -= alpha * partial_CW + momentum * W2
        b_cp -= alpha * partial_Cb + momentum * b2

        if i % 100 == 0:
            print "Iteration: ", i
            curr_W, curr_b = W_cp.copy(), b_cp.copy()
            # print "=======In while======="
            # print curr_W, curr_b
            results.append([i, curr_W, curr_b])

    return results, W_cp, b_cp


def part5():

    np.random.seed(1)
    W = np.random.randn(784, 10) * 10e-5
    b = np.zeros((10, 1))

    training = np.empty((784, 0))
    tests = np.empty((784, 0))
    y_training = np.empty((10, 0))
    y_test = np.empty((10, 0))

    for i in range(0, 10):
        training = np.hstack((training, M["train"+str(i)].T/255.0))
        tests = np.hstack((tests, M["test"+str(i)].T/255.0))
        training_size = len(M["train"+str(i)])
        test_size = len(M["test"+str(i)])
        # one hot vector
        o_vector = np.zeros((10, 1))
        o_vector[i] = 1
        y_training = np.hstack((y_training, tile(o_vector, (1, training_size))))
        y_test = np.hstack((y_test, tile(o_vector, (1, test_size))))

    # calculating performance
    alpha = 0.00001
    momentum = 0.99
    plot_data, end_W, end_b = grad_descent_m(training, y_training, W, b, alpha, 3000, momentum)

    print "===========plot data============"

    train_accuracy, test_accuracy, itr_idx = [], [], []
    for i in plot_data:
        itr_idx.append(i[0])
        curr_w, curr_b = i[1], i[2]
        training_size, test_size = training.shape[1], tests.shape[1]

        train_correct, test_correct = 0, 0
        train_x = part2(training, curr_w, curr_b)
        test_x = part2(tests, curr_w, curr_b)

        for j in range(training_size):
            if y_training[:, j].argmax() == train_x[:, j].argmax():
                train_correct += 1
        for k in range(test_size):
            if y_test[:, k].argmax() == test_x[:, k].argmax():
                test_correct += 1

        train_accuracy.append(float(train_correct)/float(training_size))
        test_accuracy.append(float(test_correct)/float(test_size))


    plt.figure()
    plt.plot(itr_idx, train_accuracy, color='blue', marker='o', label="Training")
    plt.plot(itr_idx, test_accuracy, color='green', marker='o', label="Test")
    plt.legend(loc="best")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Performance on Accuracy")
    plt.savefig('part5_learning_curves.png')

    return plot_data, [end_W, end_b]

# part5()
# =========================== PART 6 ==========================================
def grad_descent_compare(x, y, W, b, alpha, itr, momentum, if_m):
    EPS = 1e-5
    prev_w = W - 10 * EPS
    W_cp = W.copy()
    b_cp = b.copy()
    i = 0
    results = []
    while norm(W_cp - prev_w) > EPS and i < itr:
        i += 1
        prev_w = W_cp.copy()
        partial_CW, partial_Cb = df(x, y, W_cp, b_cp)

        W2, b2 = W.copy(), b.copy()
        if if_m:
            W_cp -= alpha * partial_CW + momentum * W2
            b_cp -= alpha * partial_Cb + momentum * b2
        else:
            W_cp -= alpha * partial_CW
            b_cp -= alpha * partial_Cb

        curr_W, curr_b = W_cp.copy(), b_cp.copy()
        results.append([i, curr_W, curr_b])
        if i % 100 == 0:
            print "Iteration: ", i
            #curr_W, curr_b = W_cp.copy(), b_cp.copy()
            #results.append([i, curr_W, curr_b])

    return results, W_cp, b_cp

def part6a():
    np.random.seed(1)
    W = np.random.randn(784, 10) * 10e-5
    b = np.zeros((10, 1))

    training = np.empty((784, 0))
    tests = np.empty((784, 0))
    y_training = np.empty((10, 0))
    y_test = np.empty((10, 0))

    for i in range(0, 10):
        training = np.hstack((training, M["train"+str(i)].T/255.0))
        tests = np.hstack((tests, M["test"+str(i)].T/255.0))
        training_size = len(M["train"+str(i)])
        test_size = len(M["test"+str(i)])
        # one hot vector
        o_vector = np.zeros((10, 1))
        o_vector[i] = 1
        y_training = np.hstack((y_training, tile(o_vector, (1, training_size))))
        y_test = np.hstack((y_test, tile(o_vector, (1, test_size))))

    alpha = 0.00005
    momentum = 0.99
    part5_r, part5_end_W, part5_end_b = grad_descent_compare(training, y_training, W, b, alpha, 3000, momentum, True)

    loc1, loc2 = (380, 0), (381, 0)
    # w1rs, w2rs = part5_end_W[(375,)], part5_end_W[(385,)]

    # np.random.seed(3)
    # w1rs = np.random.uniform(0, 2, 10) * 10e-5
    # w2rs = np.random.uniform(0, 2, 10) * 10e-5
    w1rs = np.arange(-0, 3, 0.05)
    w2rs = np.arange(-0, 3, 0.05)
    w1z, w2z = np.meshgrid(w1rs, w2rs)
    c = np.zeros([w1rs.size, w2rs.size])

    #b.shape # itr 200: (7840, 60000)
    #w1z, w2z = np.meshgrid(w1rs, w2rs)

    for i, w1 in enumerate(w1rs):
        for j, w2 in enumerate(w2rs):
            weights_cp = part5_end_W.copy()
            weights_cp[loc1], weights_cp[loc2] = w1, w2
            c[j, i] = cf(training, y_training, weights_cp, part5_end_b)

    plt.figure()
    plt.contour(w1z, w2z, c, cmap="coolwarm")
    plt.title("Contour Plot")
    plt.legend(loc="best")
    plt.savefig("part6a_contour.png")
    # cPickle.dump(fig_object, file("part6a.pickle", 'w'))

#part6a()

def part6b():
    np.random.seed(1)
    W = np.random.randn(784, 10) * 10e-5
    b = np.zeros((10, 1))

    training = np.empty((784, 0))
    tests = np.empty((784, 0))
    y_training = np.empty((10, 0))
    y_test = np.empty((10, 0))

    for i in range(0, 10):
        training = np.hstack((training, M["train"+str(i)].T/255.0))
        tests = np.hstack((tests, M["test"+str(i)].T/255.0))
        training_size = len(M["train"+str(i)])
        test_size = len(M["test"+str(i)])
        # one hot vector
        o_vector = np.zeros((10, 1))
        o_vector[i] = 1
        y_training = np.hstack((y_training, tile(o_vector, (1, training_size))))
        y_test = np.hstack((y_test, tile(o_vector, (1, test_size))))

    loc1, loc2 = (380, 0), (381, 0)
    alpha = 0.0005
    momentum = 0.99
    part5_r, part5_end_W, part5_end_b = grad_descent_compare(training, y_training, W, b, alpha, 20, momentum, False)

    w1rs, w2rs, gd_traj = [], [], [(0, 0)]
    for i in part5_r:
        w1rs.append(i[1][loc1])
        w2rs.append(i[1][loc2])
        gd_traj.append((i[1][loc1], i[1][loc2]))

    w1z, w2z = np.meshgrid(w1rs, w2rs)
    c = np.zeros([w1rs.size, w2rs.size])

    for i, w1 in enumerate(w1rs):
         for j, w2 in enumerate(w2rs):
              weights_cp = part5_end_W.copy()
              weights_cp[loc1], weights_cp[loc2] = w1, w2
              c[j, i] = cf(training, y_training, weights_cp, part5_end_b)

    plt.figure()
    plt.contour(w1z, w2z, c, cmap="coolwarm")
    plt.plot([a for a, b in gd_traj], [b for a, b in gd_traj], 'yo-', label="No Momentum")
    plt.title("Trajectory Plot")
    plt.legend(loc="best")
    plt.savefig("part6b.png")

#part6b()

def part6c():
    np.random.seed(1)
    W = np.random.randn(784, 10) * 10e-5
    b = np.zeros((10, 1))

    training = np.empty((784, 0))
    tests = np.empty((784, 0))
    y_training = np.empty((10, 0))
    y_test = np.empty((10, 0))

    for i in range(0, 10):
        training = np.hstack((training, M["train"+str(i)].T/255.0))
        tests = np.hstack((tests, M["test"+str(i)].T/255.0))
        training_size = len(M["train"+str(i)])
        test_size = len(M["test"+str(i)])
        # one hot vector
        o_vector = np.zeros((10, 1))
        o_vector[i] = 1
        y_training = np.hstack((y_training, tile(o_vector, (1, training_size))))
        y_test = np.hstack((y_test, tile(o_vector, (1, test_size))))

    loc1, loc2 = (380, 0), (381, 0)
    alpha = 0.0005
    momentum = 0.99
    part5_r, part5_end_W, part5_end_b = grad_descent_compare(training, y_training, W, b, alpha, 20, momentum, True)

    w1rs, w2rs, mo_traj = [], [], [(0, 0)]
    for i in part5_r:
        w1rs.append(i[1][loc1])
        w2rs.append(i[1][loc2])
        mo_traj.append((i[1][loc1], i[1][loc2]))

    w1z, w2z = np.meshgrid(w1rs, w2rs)
    c = np.zeros([w1rs.size, w2rs.size])

    for i, w1 in enumerate(w1rs):
        for j, w2 in enumerate(w2rs):
            weights_cp = part5_end_W.copy()
            weights_cp[loc1], weights_cp[loc2] = w1, w2
            c[j, i] = cf(training, y_training, weights_cp, part5_end_b)

    plt.figure()
    plt.contour(w1z, w2z, c, cmap="coolwarm")
    plt.plot([a for a, b in mo_traj], [b for a, b in mo_traj], 'go-', label="With Momentum")
    plt.title("Trajectory Plot")
    plt.legend(loc="best")
    plt.savefig("part6c.png")

#part6c()

def part6e():
    np.random.seed(1)
    W = np.random.randn(784, 10) * 10e-5
    b = np.zeros((10, 1))

    training = np.empty((784, 0))
    tests = np.empty((784, 0))
    y_training = np.empty((10, 0))
    y_test = np.empty((10, 0))

    for i in range(0, 10):
        training = np.hstack((training, M["train"+str(i)].T/255.0))
        tests = np.hstack((tests, M["test"+str(i)].T/255.0))
        training_size = len(M["train"+str(i)])
        test_size = len(M["test"+str(i)])
        # one hot vector
        o_vector = np.zeros((10, 1))
        o_vector[i] = 1
        y_training = np.hstack((y_training, tile(o_vector, (1, training_size))))
        y_test = np.hstack((y_test, tile(o_vector, (1, test_size))))

    loc1, loc2 = (100, 0), (600, 0)
    alpha = 0.0005
    momentum = 0.99
    part5_r, part5_end_W, part5_end_b = grad_descent_compare(training, y_training, W, b, alpha, 3000, momentum, True)

    gd_traj, mo_traj = [], []
    for i in part5_r:
        gd_traj.append((i[1][loc1], i[1][loc2]))
        mo_traj.append((i[1][loc1], i[1][loc2]))

    w1rs = np.arange(1, 4, 0.05)
    w2rs = np.arange(1, 4, 0.05)
    w1z, w2z = np.meshgrid(w1rs, w2rs)
    c = np.zeros([w1rs.size, w2rs.size])
    #b.shape # itr 200: (7840, 60000)

    for i, w1 in enumerate(w1rs):
        for j, w2 in enumerate(w2rs):
            weights_cp = part5_end_W.copy()
            weights_cp[loc1], weights_cp[loc2] = w1, w2
            c[j, i] = cf(training, y_training, weights_cp, part5_end_b)

    plt.figure()
    plt.contour(w1z, w2z, c, cmap="coolwarm")
    plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    plt.title("Contour Plot")
    plt.legend(loc="best")
    plt.savefig("part6e.png")

#part6e()

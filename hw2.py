import numpy as np
import matplotlib.pyplot as plt
import random


def generateData(sample_size):
    points = np.matrix(np.random.uniform(-1, 1.00000001, size=(
        sample_size, 1,
        2)))  # since the function excludes the maximum bound while generating ,i have kept the maximum bound
    hypothesis_points = np.matrix(
        np.random.uniform(-1, 1, size=(2, 1, 2)))  # selecting 2 points for uniform distribution
    slope = (hypothesis_points[0, 1] - hypothesis_points[1, 1]) / (
        hypothesis_points[0, 0] - hypothesis_points[1, 0])  # calculating slope y2-y1/x2-x1
    # y-y1 - m(x-x1) = 0 is the line, this line divides the whole plane into two parts, point
    x_coordinates = points[0:, 0:1]
    y_coordinates = points[0:, 1:2]
    # taking first point in hypothesis
    y1 = hypothesis_points[0:1, 1:2]

    x1 = hypothesis_points[0:1, 0:1]

    # using linear inequality dividing the points with -1 and +1 values
    # labelling the input matrix into +1 or -1 class
    label = np.sign((y_coordinates - y1) - (slope * (x_coordinates - x1)))

    return points, label


def pseudoinverse(points, labels):
    # preparing artifical coordinate for adding it to the input matrix
    x0_matrix = np.ones((points.shape[0], 1))

    # each row consists of point, adding x0 artificial coordinate for every point
    x_matrix = np.matrix(np.concatenate((x0_matrix, points), axis=1))
    # calculating transpose
    x_tranmat = np.transpose(x_matrix)
    # finding out the pseudo inverse  and multiplying with y and finding transpose
    weight = np.transpose((np.linalg.inv(x_tranmat * x_matrix) * x_tranmat) * labels)

    return weight


def pla(points, labels, intial_weight_vector):
    # creating artifical coordinate matrix for concatenating with input matrix
    x0_matrix = np.ones((points.shape[0], 1))

    # each row consists of point, adding x0 artificial coordinate for every point
    x_matrix = np.matrix(np.concatenate((x0_matrix, points), axis=1))
    # making a matrix of intial weight vectors
    intial_weight_vector = np.matrix(np.tile(intial_weight_vector, (points.shape[0], 1)))
    # calculating y values intially
    h_x = np.sign(np.sum((np.multiply(intial_weight_vector, x_matrix)), axis=1))
    weight = intial_weight_vector

    no_of_iterations = 0
    # running the loop till all the y values are matching with label matrix
    while np.array_equal(h_x, labels) == bool(False):
        unequal = h_x - labels  # all equal elements willl be turned to zero,unequal elements transform into non zero elements
        misclassifiedpoints = np.flatnonzero(np.array(unequal))# picking non zero elements indices
        random_misclassifiedpoint = random.choice(misclassifiedpoints)#selecting one random point
        #updating the weight matrix
        weight = weight + (labels[random_misclassifiedpoint, 0] * x_matrix[random_misclassifiedpoint:random_misclassifiedpoint + 1, :])

        # finding the  labels for new weights
        h_x = np.sign(np.sum((np.multiply(weight, x_matrix)), axis=1))
        no_of_iterations += 1

    # slicing the weight vector from weight matrix
    weight = weight[0:1, :]
    return weight, no_of_iterations


# validating and comparing the results

def validate(x_input, label, pla_weights, lr_weights):
    x0_matrix = np.ones((x_input.shape[0], 1))

    # each row consists of point, adding x0 artificial coordinate for every point
    x_matrix = np.matrix(np.concatenate((x0_matrix, x_input), axis=1))
    # making a matrix of intial weight vectors
    pla_weights = np.matrix(np.tile(pla_weights, (x_input.shape[0], 1)))
    lr_weights = np.matrix(np.tile(lr_weights, (x_input.shape[0], 1)))
    # calculating y values intially
    pla_labels = np.sign(np.sum((np.multiply(pla_weights, x_matrix)), axis=1))
    lr_labels = np.sign(np.sum((np.multiply(lr_weights, x_matrix)), axis=1))

    if np.array_equal(pla_labels, label) & np.array_equal(lr_labels, label):
        pass
    else:
        print 'not okay check the working'

# N = int(raw_input('Enter number of points '))
N = [10,50,100,200,500,1000]
for every_sample in N:
    plaiter = 0 #STORES COUNT OF ITERATIONS AS SUM  FOR PLA
    lriter = 0#STORES COUNT OF ITERATIONS AS SUM  FOR PLA WITH PSEUDO INV


    for iterate in range(100):  # running each trial 100 times
        # generating random points within the bounds
        [X, Y] = generateData(every_sample)
        # creating zerop weight matrix for PLA
        w0 = np.matrix(np.zeros((1, 3)))  # w_0 w_1 w_2
        # finding the weight and no of iterations
        [w, iters] = pla(X, Y, w0)
        # total no  of iterations for all the repetitions of pla
        plaiter += iters
        weight_with_pla = w
        # calling pseudo inverse  function for finding the intial weights

        # 2nd algorithm
        [w] = pseudoinverse(X, Y)
        # giving the input weights to pla

        [w, iters] = pla(X, Y, w)
        weight_with_lr_intial_weights = w
        # sum  of iterations for  repetitions of pla with solved intial weights
        lriter += iters
        # validation function optional
        #validate(X, Y, weight_with_pla, weight_with_lr_intial_weights)

    print 'average iteration for pla  with %d' % every_sample
    print plaiter / 100

    print 'average iteration for linear regression  with %d' % every_sample
    print lriter / 100

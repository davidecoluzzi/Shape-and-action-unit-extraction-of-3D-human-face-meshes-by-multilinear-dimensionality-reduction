# HOSVD

# import of the libraries
import csv
import numpy as np
from tensorly.decomposition import tucker
from tensorly import unfold
import tensorly.tenalg.n_mode_product as n_mode
from scipy import linalg
import operator
import logging
import time
import numpy as np
import math as math
import matplotlib.pyplot as plt

# function to create the tensor starting from the acquired data
def create_tensor(): 
    file_count = 0
    user_count = 0
    expr_count = 0
    riga = 0;
    colonna = 0;
    exp_name = ''
    matrix = np.zeros((20, 18, 4041))
    row_array = []

    # loop to open the data
    while True:
        if(user_count + 1 == 4):
            user_count = user_count + 1

        if expr_count == 0:
            exp_name = 'felice'
        if expr_count == 3:
            exp_name = 'triste'
        if expr_count == 6:
            exp_name = 'spaventato'
        if expr_count == 9:
            exp_name = 'arrabbiato'
        if expr_count == 12:
            exp_name = 'disgustato'
        if expr_count == 15:
            exp_name = 'sorpreso'
        if expr_count == 18:
            user_count = user_count + 1
            if (user_count + 1 == 4):
                user_count = user_count + 1
            expr_count = 0;
            exp_name = 'felice'

        if user_count == 21:
            break;

        count_mod = operator.mod(expr_count, 3)
        user_name = ('u' + str(user_count + 1) + '_' + exp_name + '_' + str(count_mod)) # create the name
        path = 'Matlab/Dataset_Csv/' + user_name + '.csv'
        expr_count = expr_count + 1

        try:
            csv = np.genfromtxt(path, delimiter=",") # open the file
            file_count = file_count + 1;
            one_line = np.array([])
            # format (x, y, z) of the cloud points in an array of 4041 elements
            for i in range(0, 1347):
                for j in range(0, 3):
                    one_line = np.append(one_line, csv[j, i])

            matrix[riga][colonna] = one_line # store the tensor
            colonna = colonna + 1
            if colonna == 18:
                riga = riga + 1
                colonna = 0

        except IOError:
            print("File not found")

    return matrix

# function to compute a mean face, save it and center the tensor data
def center_tensor(orTen): 

    # compute the sum of each point for each face
    sum = np.zeros(4041)
    for i in range(0, 20):
        for y in range(0, 18):
            for j in range(0, 4041):
                sum[j] = sum[j] + orTen[i, y, j]

    meanVec = np.zeros(4041)
    meanMat = np.zeros((3, 1347))

    # divide the sum (of each element) by the number of faces that have been added
    numEl = 20 * 18
    for i in range(0, 4041):
        meanVec[i] = sum[i] / numEl

    # creation of the mean face from the obtained array and saving in a .csv file
    x = 0
    for i in range(0, 4041):
        y = operator.mod(i, 3)
        if y == 0 and i != 0:
            x = x + 1
        meanMat[y, x] = meanVec[i]

    np.savetxt('MeanFace.csv', meanMat, fmt='%.4e', delimiter=',')

    # centering the data subtracting the mean face
    allTen = np.zeros((20, 18, 4041))
    for i in range(0, 20):
        for y in range(0, 18):
            allTen[i][y] = orTen[i][y] - meanVec

    # reshape the centered tensor in 4041x20x18
    allTen2 = np.zeros((4041, 20, 18))
    for i in range(0, 20):
        for y in range(0, 18):
            for j in range(0, 4041):
                allTen2[j][i][y] = allTen[i][y][j]

    print("Centered tensor")
    print(allTen2.shape)

    return allTen2, meanVec

# function to compute maximum and minimum and save it in a .csv file
def compute_maxMin(mode2, mode3):
    # compute maximum and minimum for each value of each of the two matrices of factors and save them in a matrix 2x16.
    # In this, in the first line, there are consecutively maximum and minimum for each unit for the identity space and in the
    # second the corresponding ones for the expression space

    maxmin = np.zeros((2, 16))

    for y in range(0, 2):
        for i in range(0, 16, 2):
            j = i / 2
            maxmin[0][i] = max(mode2[int(j), :])
            maxmin[0][i + 1] = min(mode2[int(j), :])
            maxmin[1][i] = max(mode3[int(j), :])
            maxmin[1][i + 1] = min(mode3[int(j), :])

    np.savetxt('MaxMin.csv', maxmin, fmt='%.4e', delimiter=',')

# function to reconstruct a particular face and the result is written to a file
def reconstruct_face(cor, mode2, mode3): # Example of reconstruction
    
    # the shape and action units of a particular face are chosen from the matrices obtained with the SVD
    print("Shape/action units of the example face")
    coeffSh = np.array(mode2[9, :])[np.newaxis]  # identity number 10
    coeffShTransp = np.array(coeffSh.T)
    print(coeffShTransp.shape)
    coeffEx = np.array(mode3[14, :])[np.newaxis]  # expression surprised with 2 grade of emphasis
    coeffExTransp = np.array(coeffEx.T)
    print(coeffExTransp.shape)

    prod2 = (n_mode.mode_dot(core, coeffSh, 1)) # 2-mode product
    prod3 = (n_mode.mode_dot(prod2, coeffEx, 2)) # 3-mode product

    prodFac = np.zeros(4041)
    for i in range(0, 4041):
        prodFac[i] = prod3[i][0][0]

    faccia = meanVec + prodFac  # reconstruction of the face adding the result of the n-mode products to the mean face
    faccia1 = np.zeros((3, 1347)) # reshape the obtained model of points in a matrix of 3x1347
    x = 0
    for i in range(0, 4041):
        y = operator.mod(i, 3)
        if (y == 0 and i != 0):
            x = x + 1;
        faccia1[y, x] = faccia[i]

    print("Reconstruction of the example face: ")
    print(faccia1)

    np.savetxt('Face1.csv', faccia1, fmt='%.4e', delimiter=',')

# plot the singular values and the cumulative percentage
def plot_test(allTen2): 

    m1 = unfold(allTen2, 1) # 2 mode of the tensor
    m2 = unfold(allTen2, 2) # 3 mode of the tensor

    s1 = linalg.svd(m1, compute_uv=False);  # compute the singular values of the second tensor mode
    s2 = linalg.svd(m2, compute_uv=False);  # compute the singular values of the third tensor mode 

    print("Singular values obtained from the SVD of the second tensor mode")
    print(s1)
    print("Singular values obtained from the SVD of the third tensor mode")
    print(s2)

    n_s1 = np.zeros(len(s1)) # values on the x-axis 
    for i in range(1, len(s1) + 1):
        n_s1[i - 1] = i

    plt.plot(n_s1, s1)
    plt.show()

    n_s2 = np.zeros(len(s2)) # values on the x-axis
    for i in range(1, len(s2) + 1):
        n_s2[i - 1] = i

    plt.plot(n_s2, s2)
    plt.show()

    # percentage of the variance (cumulative)
    sum1 = 0
    for i in range(0, len(s1)):
        sum1 += s1[i]

    s1_p = np.zeros(len(s1))
    for i in range(0, len(s1)):
        s1_p[i] = (s1[i] * 100) / sum1 + s1_p[i - 1]

    plt.plot(n_s1, s1_p)
    plt.show()

    sum2 = 0
    for i in range(0, len(s2)):
        sum2 += s2[i]

    s2_p = np.zeros(len(s2))
    for i in range(0, len(s2)):
        s2_p[i] = (s2[i] * 100) / sum2 + s2_p[i - 1]

    plt.plot(n_s2, s2_p)
    plt.show()

if __name__ == '__main__':

    orTen = create_tensor() # creation of the tensor
    allTen2, meanVec = center_tensor(orTen) # data centering

    nn_core, nn_factors = tucker(allTen2, ranks = [4041, 8, 8]) # SVD of the tensor modes

    print("Factors")
    print("Shape")
    mode2 = nn_factors[1] # eigenvectors obtained from the SVD of tensor mode 2
    print(mode2.shape)
    print("Expression")
    mode3 = nn_factors[2] # eigenvectors obtained from the SVD of tensor mode 3
    print(mode3.shape)

    compute_maxMin(mode2, mode3) # compute maximum and minimum of the factor matrices

    core = np.zeros((4041, 8, 8)) # core computation

    prod2 = (n_mode.mode_dot(allTen2, mode2.T, 1))
    core = (n_mode.mode_dot(prod2, mode3.T, 2))

    # before the storage on file, it is necessary the unfolding, no tensor saving on a .csv file
    # before using it, it will be necessary to do th folding 
    core1 = unfold(core, 1)
    np.savetxt('Core.csv', core1, fmt='%.4e', delimiter=',')

    reconstruct_face(core, mode2, mode3) # face reconstruction example 

    plot_test(allTen2) # plot of the singular values and the cumulative percentage 

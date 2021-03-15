import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


data = pd.read_csv('function2_2d.csv')
x1 = data['x1'].to_numpy()
x2 = data['x2'].to_numpy()
y = data['y'].to_numpy()


def get_phi(x1, x2, degree=2):
    phi = []
    for i in range(x1.shape[0]):
        curr = []
        curr_x1 = x1[i]
        curr_x2 = x2[i]
        curr.append(1)
        for j in range(1, degree + 1):
            num_terms = j + 1
            for k in range(num_terms):
                pow_x1 = math.pow(curr_x1, k)
                pow_x2 = math.pow(curr_x2, j - k)
                curr.append(pow_x1 * pow_x2)
        phi.append(curr)
    return np.array(phi)


def predict(x, weights):
    return np.matmul(x, weights)


def fit(x1, x2, y, degree=2, lambda_=0.0):
    phi = get_phi(x1, x2, degree=degree)
    num_terms = 0
    for i in range(degree + 1):
        num_terms += i + 1
    w1 = np.matmul(phi.T, phi) + lambda_ * np.identity(num_terms)
    w2 = np.matmul(phi.T, y)
    weights = np.matmul(np.linalg.inv(w1), w2)
    return weights


def test(weights, x1_, x2_, degree=2, num_points=50):
    phi = get_phi(x1_, x2_, degree=degree)
    y_test = np.matmul(phi, weights)
    return x1_, x2_, y_test


DATASET_SIZE = 500
DEGREE = 6
res = fit(x1[0:DATASET_SIZE], x2[0:DATASET_SIZE], y[0:DATASET_SIZE], degree=DEGREE, lambda_=1)
x1_, x2_,  y_ = test(res, x1[0:DATASET_SIZE], x2[0:DATASET_SIZE], degree=DEGREE)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x1_, x2_, y_)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('function2.csv')
x = data['x'].to_numpy()
y = data['y'].to_numpy()

DEGREE = 2


def get_polynomial_data(x_data, degree=1):
    poly_data = []
    for a in x_data:
        curr = []
        curr_val = 1
        for i in range(degree + 1):
            curr.append(curr_val)
            curr_val *= a
        poly_data.append(curr)
    return np.array(poly_data)


def normalize_data(x):
    x[:, 1:] = (x[:, 1:] - np.mean(x[:, 1:], axis=0)) / np.std(x[:, 1:], axis=0)
    return x


def predict(x, weights):
    return np.matmul(x, weights)


def compute_error(x, y, weights):
    preds = np.matmul(x, weights)
    err = np.subtract(preds, y)
    return err


def fit(x_data, y_data, degree=1, lambda_=0.1):
    phi = get_polynomial_data(x_data, degree=degree)
    x1 = np.matmul(phi.T, phi) + lambda_ * np.identity(degree + 1)
    x2 = np.matmul(phi.T, y_data)
    weights = np.matmul(np.linalg.inv(x1), x2)
    return weights


def train(x_data, y_data, degree=1, num_iters=100, learning_rate=0.1):
    x_data = get_polynomial_data(x_data, degree=degree)
    x_data = normalize_data(x_data)
    weights = np.random.randn(degree + 1)
    for i in range(num_iters):
        err = compute_error(x_data, y_data, weights)
        grads = np.matmul(x_data.T, err)
        weights -= learning_rate * grads
    return weights


def test(weights, degree=2, num_points=50):
    x_test = np.linspace(-1.0, 1.0, num=num_points)
    x_test_poly = get_polynomial_data(x_test, degree)
    x_test_poly = normalize_data(x_test_poly)
    y_test = np.matmul(x_test_poly, weights)
    return x_test, y_test


DATASET_SIZE = 10
degrees = [2, 3, 6, 9]
lambdas = [0.0, 0.01, 0.1, 1.0]
fig, ax = plt.subplots()
for lambda_ in lambdas:
    res = fit(x[0:DATASET_SIZE], y[0:DATASET_SIZE], 9, lambda_=lambda_)
    x_test, y_test = test(res, degree=9, num_points=50)
    ax.plot(x_test, y_test, label='Lambda=' + str(lambda_))
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


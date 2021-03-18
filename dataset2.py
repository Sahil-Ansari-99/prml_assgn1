import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


data = pd.read_csv('function2_2d.csv')
x1 = data['x1'].to_numpy()
x2 = data['x2'].to_numpy()
y = data['y'].to_numpy()
data = data.drop(['y'], axis=1)
data = data.drop(['Unnamed: 0'], axis=1)
x = data.to_numpy()
print(x.shape)


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


def quadratic_regularization(phi, y, lambda_=0.0):
    w1 = np.matmul(phi.T, phi) + lambda_ * np.identity(phi.shape[1])
    w2 = np.matmul(phi.T, y)
    weights = np.matmul(np.linalg.inv(w1), w2)
    return weights


def tikhonov_regularization(phi, y, mus, sigma=1, lambda_=0.0):
    phi_prime = np.zeros((phi.shape[1], phi.shape[1]))
    den = sigma ** 2
    for i in range(mus.shape[0]):
        for j in range(mus.shape[0]):
            if i == j:
                phi_prime[i, j] = 1
            else:
                phi_prime[i, j] = np.exp(-np.linalg.norm(mus[i] - mus[j]) ** 2 / den)
    w1 = np.matmul(phi.T, phi) + lambda_ * phi_prime
    w2 = np.matmul(phi.T, y)
    weights = np.matmul(np.linalg.inv(w1), w2)
    return weights


def test(weights, x1_, x2_, degree=2, num_points=50):
    phi = get_phi(x1_, x2_, degree=degree)
    y_test = np.matmul(phi, weights)
    return x1_, x2_, y_test


def compute_error(phi, weights, y):
    error = 0.0
    for i in range(phi.shape[0]):
        point = phi[i]
        pred = np.matmul(point, weights)
        error += (y[i] - pred) ** 2
    return (2 * error / phi.shape[0]) ** 0.5


def get_kmeans(k, x, num_iters=2):
    init_means = np.random.randint(0, x.shape[0], size=k)
    kmeans = np.zeros((k, x.shape[1]))
    for i in range(k):
        kmeans[i] = x[init_means[i]]
    point_means = np.zeros(x.shape)
    tol = 0.001
    for i in range(num_iters):
        for j in range(x.shape[0]):
            point = x[j]
            min_dist = 999
            curr_mean = kmeans[0]
            for k in range(kmeans.shape[0]):
                curr_dist = np.linalg.norm(point - kmeans[k])
                if curr_dist < min_dist:
                    curr_mean = kmeans[k]
                    min_dist = curr_dist
            point_means[j] = curr_mean
        err = 0
        for j in range(kmeans.shape[0]):
            curr_sum = np.zeros(x.shape[1])
            curr_num = 0
            for k in range(x.shape[0]):
                if (point_means[k] == kmeans[j]).all():
                    curr_sum += x[k]
                    curr_num += 1
            new_mean = curr_sum / curr_num
            err += np.linalg.norm(kmeans[j] - new_mean)
            kmeans[j] = new_mean
        if err < tol:
            break
    sigma = 0
    for j in range(kmeans.shape[0]):
        curr_sum = 0
        curr_count = 0
        for k in range(x.shape[0]):
            if (point_means[k] == kmeans[j]).all():
                curr_sum += np.linalg.norm(x[k] - kmeans[j]) ** 2
                curr_count += 1
        sigma += curr_sum
    return kmeans, sigma / x.shape[0]


def get_gaussian_phi(kmeans, x, sigma):
    phi = []
    den = sigma ** 2
    for point in x:
        curr = []
        for mu in kmeans:
            curr.append(np.exp(-(np.linalg.norm(point - mu) ** 2) / den))
        phi.append(curr)
    return np.array(phi)


# DATASET_SIZE = 500
# DEGREE = 6
# res = fit(x1[0:DATASET_SIZE], x2[0:DATASET_SIZE], y[0:DATASET_SIZE], degree=DEGREE, lambda_=1)
# x1_, x2_,  y_ = test(res, x1[0:DATASET_SIZE], x2[0:DATASET_SIZE], degree=DEGREE)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(x1_, x2_, y_)
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('y')
# plt.show()
DATASET_SIZE = x.shape[0]
TRAIN_SIZE = int(0.7 * DATASET_SIZE)
VAL_SIZE = int(0.2 * DATASET_SIZE)
TEST_SIZE = int(0.1 * DATASET_SIZE)
mean_sizes = [10, 15, 21, 25, 30]
lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0]
sigmas = [8, 9, 18, 20, 22, 30]
rmse = []
for k in mean_sizes:
    kmeans, _ = get_kmeans(k, x[0:TRAIN_SIZE], num_iters=100)
    phi = get_gaussian_phi(kmeans, x, 20)
    weights = tikhonov_regularization(phi[0:TRAIN_SIZE], y[0:TRAIN_SIZE], kmeans, 20, lambda_=0.1)
    err = compute_error(phi[0:TRAIN_SIZE], weights, y[0:TRAIN_SIZE])
    rmse.append(err)
    print('K:', k, 'lambda:', 0.0001, 'sigma:', 20)
    print('Pred:', np.matmul(phi[100], weights))
    print('Actual:', y[100])
    print('Error:', err)

fig, ax = plt.subplots()
ax.plot(mean_sizes, rmse)
plt.xlabel('Cluster Size')
plt.ylabel('Erms')
plt.show()

# kmeans, _ = get_kmeans(100, x, num_iters=100)
# phi = get_gaussian_phi(kmeans, x, 8)
# weights = quadratic_regularization(phi[0:TRAIN_SIZE], y[0:TRAIN_SIZE], lambda_=0.0001)
# preds = np.matmul(phi[0:TRAIN_SIZE], weights)
#
# fig, ax = plt.subplots()
# ax.scatter(y[0:TRAIN_SIZE], preds)
# plt.xlabel('Target Output')
# plt.ylabel('Model Output')
# plt.show()

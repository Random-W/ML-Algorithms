import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
Program written to perform linear regression using gradient descent.
Result visualization available for 2D and 3D regression.
Data read in from .txt files with samples with the following format, 
x1, x2, ..., xn, y
x1, x2, ..., xn, y
...
'''


# parameters:
# x - inputs, m*(n+1)
# output:
# x_norm - normalized inputs, m*(n+1)
# avg - average of all features, (n+1)
# std - standard deviation of feature, (n+1)
def normalize(x):
    std = np.std(x, 0)
    avg = np.average(x, 0)
    x_norm = (x - avg)/std
    return x_norm, avg, std


# parameters:
# theta - weights, (n+1)*1
# x - inputs, m*(n+1)
# y - outputs, m*1
# output:
# j - cost calculated using squared error
def calc_cost(theta, x, y):
    m = np.size(raw, 0)
    j = 1/2/m*np.sum((np.dot(x, theta)-y)**2)
    return j


# parameters:
# theta - weights, (n+1)*1
# x - inputs, m*(n+1)
# y - outputs, m*1
# output:
# gradient - gradient at theta, (n+1)*1
def calc_gradient(theta, x, y):
    m = np.size(raw, 0)
    gradient = 1/m*np.sum(np.multiply(np.transpose(x), np.transpose(np.dot(x, theta)-y)), 1)
    return gradient.reshape((np.size(gradient),1))


# parameters:
# theta - weights, (n+1)*1
# x - inputs, m*(n+1)
# y - outputs, m*1
# plot_cost - whether to plot cost against iterations
# output:
# theta - final weights , (n+1)*1
def gradient_descent(theta, x, y, plot_cost):
    iterations = 1500
    alpha = 0.01
    cost = np.zeros(iterations)
    for i in range(iterations):
        cost[i] = calc_cost(theta, x, y)
        theta = theta - alpha*calc_gradient(theta, x, y)
    if plot_cost:
        plt.plot(range(iterations),cost)
        plt.show()
    return theta


# parameters:
# theta - weights, (n+1)*1
# x - inputs, m*(n+1)
# y - outputs, m*1
# avg - average of features before normalization, (n+1)
# std - standard deviation of feature before normalization, (n+1)
# output:
# No output
def plot_result(theta, x, y, avg, std):
    theta[0][0] -= np.dot(np.divide(avg,std),theta[1:, 0])
    theta[1:,0] = np.divide(theta[1:, 0], std)
    x = np.multiply(x[:, 1:], std)+avg
    if np.size(theta) == 2:
        plt.plot(x, y, 'r*')
        plt.plot([0, max(x)], [theta[0][0],theta[0][0]+max(x)*theta[1][0]])
    else:
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        # plot data points
        ax.scatter(x[:, 0], x[:, 1], y)
        # plot surface
        x1 = np.arange(min(x[:, 0]), max(x[:, 0]),(max(x[:, 0])-min(x[:, 0]))/500)
        x2 = np.arange(min(x[:, 1]), max(x[:, 1]),(max(x[:, 1])-min(x[:, 1]))/500)
        x1, x2 = np.meshgrid(x1, x2)
        y = theta[2][0]*x2 + theta[1][0]*x1 + theta[0][0]
        ax.plot_surface(x1, x2, y, cmap=plt.cm.winter)
    plt.show()


if __name__ == "__main__":
    # read in raw data
    raw = np.genfromtxt("ex1data2.txt", dtype=np.double, delimiter=",")
    n = np.size(raw, 1) - 1
    m = np.size(raw, 0)
    # normalization
    x_norm, avg, std = normalize(raw[:, 0:n])
    # generate training samples
    x = np.hstack((np.ones((m, 1)), x_norm))
    y = raw[:, -1].reshape((m, 1))
    # initialization for gradient descent
    theta = np.zeros((n + 1, 1))
    # gradient descent
    theta = gradient_descent(theta, x, y, True)
    print(theta)
    # plot data
    if n <= 2:
        plot_result(theta, x, y, avg, std)


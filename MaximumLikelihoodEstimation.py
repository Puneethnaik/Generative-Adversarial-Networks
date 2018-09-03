import sys
sys.path.insert(0, "/home/puneeth/Projects/GAN/GenerativeAdversarialNetworks/utilities")
import mini_batch_gradient_descent as gd
import plotting_functions as pf
import numpy as np
def gaussian(X, theta):
    res = (1 / (np.sqrt(2 * np.pi * theta[1]))) * np.exp(-(X - theta[0]) ** 2 / (2 * theta[1]))
    return res
X = np.transpose(np.random.random_sample(10) * 2.0)
obj = [X]
mle = gd.GradientDescentOptimizer(X, 10**-4, 1)
theta = mle.optimize()
# print(np.random.normal(theta[0], theta[1], 5))
# print("gaussian", gaussian(X, [5, 5]))
pred = np.transpose(gaussian(X, theta))
obj.append(pred)
print(theta)
print("theta")
pf.plot_graphs(1, 5, 10, obj)
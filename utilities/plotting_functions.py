import matplotlib.pyplot as plt
from numpy import *
def plot_graphs(start, end, steps, X):
    t = linspace(start, end, steps)
    # axes = []
    plt.plot(t, X[0], 'r')
    plt.plot(t, X[1], 'g')
    plt.show()
import numpy as np
from numpy import cos, sin, pi
from scipy.linalg import det, expm, inv, sqrtm
import matplotlib.pyplot as plt
from numpy.random import randn

""" Exo 4.1 """

x_mean = np.array([[1], [2]])
gamma_x = np.eye(2)

def ddp(x, mean=x_mean, gamma=gamma_x):
    alpha = 1 / (2 * np.sqrt((2 * np.pi)**2) * det(gamma))
    d = x - mean
    M = -1/2 * d.T @ inv(gamma) @ d
    return alpha * expm(M)


""" Exo 4.2 """

# a = np.sqrt(-2 * np.log(1 - 0.9))
A1 = np.array([[1, 0], [0, 3]])
A2 = np.array([[cos(pi/4), -sin(pi/4)], [sin(pi/4), cos(pi/4)]])
I = np.eye(2)

G1 = I
G2 = 3 * G1
G3 = A1 @ G2 @ A1.T + I
G4 = A2 @ G3 @ A2.T
G5 = G3 + G4
G6 = A2 @ G5 @ A2.T

G = [G1, G2, G3, G4, G5, G6]

def eps(G, eta, mean=0):
    a = np.sqrt(-2 * np.log(1 - eta))
    theta = np.linspace(0, 2 * np.pi, 100)
    for mat in G:
        X, Y = [], []
        for th in theta:
            x, y = mean + a * sqrtm(mat) @ np.array([[np.cos(th)], [np.sin(th)]])
            X.append(x)
            Y.append(y)
        plt.plot(X, Y)


eps(G, 0.9)
# plt.axis('equal')
# plt.grid()
# plt.show()


""" Exo 4.3 """

y = randn(2, 1000)
mean = np.array([[1], [2]])
gamma = np.array([[4, 3], [3, 3]])

x = mean + sqrtm(gamma) @ y

plt.scatter(y[0], y[1], color='blue')
plt.scatter(x[0], x[1], color='red')
for eta in [0.9, 0.99, 0.999]:
    eps([gamma], eta, mean)

plt.axis('square')
plt.grid()
plt.show()


import numpy as np
from numpy import cos, sin, pi
from scipy.linalg import det, expm, inv, sqrtm
import matplotlib.pyplot as plt
from numpy.random import randn

def Q4_7():
    xbar = np.array([[0], [0]])
    y = np.array([[8],[7],[0]])
    C = np.array([[2,3],[3,2],[1,-1]])
    G_b = np.array([[1,0,0],[0,4,0],[0,0,4]])
    G_x = np.array([[1000,0],
                   [0, 1000]])

    G_y = C@G_x@C.T + G_b
    K = G_x@C.T@np.linalg.inv(G_y)
    yt = y - C@xbar
    xc = xbar + K@yt
    beta = y - C@xc
    G_eps = G_x -K@C@G_x
    print("x chapeau = ", xc)
    print("beta = ", beta)
    print("gamma epsilon = ", G_eps)

def Q4_8():
    y = np.array([5,10,8,14, 17]).reshape((5,1))
    C = np.array([[4,0],[10,1],[10,5],[13,5],[15,3]])
    G_b = 9*np.identity(5)
    G_x = 4*np.identity(2)
    xbar = np.array([[1],[-1]])
    G_y = C @ G_x @ C.T + G_b
    K = G_x @ C.T @ np.linalg.inv(G_y)
    yt = y - C @ xbar
    xc = xbar + K @ yt
    beta = y - C @ xc
    G_eps = G_x - K @ C @ G_x
    print("x chapeau = ", xc)
    print("beta = ", beta)
    print("gamma epsilon = ", G_eps)

def Q4_9():
    y = np.array([0.38,3.25,4.97, -0.26]).reshape((4,1))
    t = np.array([1,2,3,7])
    C = np.zeros((4,2))
    for i in range(4):
        C[i]=np.array([1, -np.cos(t[i])])

    G_b = np.array([[0.01, 0,0,0],[0,0.01,0,0],[0,0,0.01,0],[0,0,0,0.01]])
    G_x = 1000 * np.identity(2)
    xbar = np.array([[0], [0]])

    G_y = C @ G_x @ C.T + G_b
    K = G_x @ C.T @ np.linalg.inv(G_y)
    yt = y - C @ xbar
    xc = xbar + K @ yt
    beta = y - C @ xc
    G_eps = G_x - K @ C @ G_x
    print("x chapeau = ", xc)
    print("beta = ", beta)
    print("gamma epsilon = ", G_eps)

    p1,p2 = xc
    t = np.linspace(0,10,1000)
    x = np.zeros((1000,1))
    y = np.zeros((1000,1))
    for i in range(1000):
        x[i]= p1*t[i] -p2*np.sin(t[i])
        y[i] = p1 - p2*np.cos(t[i])
    plt.figure()
    plt.grid()
    plt.plot(x,y)
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Estimated path of the mass y(x)')





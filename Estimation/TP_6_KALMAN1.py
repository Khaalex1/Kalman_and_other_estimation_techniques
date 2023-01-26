import numpy as np
from numpy import cos, sin, pi
from scipy.linalg import det, expm, inv, sqrtm
import matplotlib.pyplot as plt
from numpy.random import randn

def kalman_correction(xc, y, C, G_x, G_b):
    yt = y - C @ xc
    G_y = C @ G_x @ C.T + G_b
    K = G_x @ C.T @ np.linalg.inv(G_y)
    xc = xc + K @ yt
    G_x = G_x - K @ C @ G_x
    return xc, G_x

def kalman_prediction(xc, G_x, A, u, alpha, G_alpha):
    xc = A@xc + u + alpha
    G_x = A@G_x@A.T + G_alpha
    return xc, G_x

def eps(G, eta, mean, lbl):
    a = np.sqrt(-2 * np.log(1 - eta))
    theta = np.linspace(0, 2 * np.pi, 100)
    #plt.figure()
    for i in range(len(G)):
        X, Y = [], []
        for th in theta:
            x, y = mean + a * sqrtm(G[i]) @ np.array([[np.cos(th)], [np.sin(th)]])
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, label = lbl)
    plt.axis('square')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ellipses de confiance à {}'.format(eta))

#cas où le système n'évolue pas : A = I, u = 0
def Q4_10():
    A = np.identity(2)
    xbar = np.array([[0], [0]])
    y = np.array([np.array([[8]]), np.array([[7]]), np.array([[0]])])
    C = np.array([[2, 3], [3, 2], [1, -1]])
    G_b = np.array([[np.array([[1]])], [np.array([[4]])], [np.array([[4]])]])
    G_x = np.array([[1000, 0],
                    [0, 1000]])
    for i in range(3):
        G_y = C[i, :] @ G_x @ C[i,:].T + G_b[i,0]

        K = (G_x @ C[i, :].T * np.linalg.inv(G_y)).reshape((2,1))

        yt = y[i] - C[i,:] @ xbar
        print(yt)
        xc = xbar + K @ yt
        beta = y[i] - C[i,:] @ xc
        G_eps = G_x - K @ np.array([C[i,:]]) @ G_x

        print("x chapeau = ", xc)
        print("gamma epsilon = ", G_eps)



    print("xchapeau = ", xc)
    print("Gamma epsilon", G_eps)

def Q4_11():
    A = [0,0,0]
    u = [0,0,0]
    C = [0,0,0]
    y = [0,0,0]
    A[0] = np.array([[0.5, 0], [0,1]])
    A[1] = np.array([[1, -1], [1, 1]])
    A[2] = np.array([[1,-1], [1,1]])
    u[0]=np.array([[8], [16]])
    u[1] = np.array([[-6],[-18]])
    u[2]=np.array([[32], [-8]])
    C[0]= np.array([[1,1]])
    C[1] = np.array([[1, 1]])
    C[2] = np.array([[1, 1]])
    y[0]=np.array([[7]])
    y[1] = np.array([[30]])
    y[2] = np.array([[-6]])
    Ga = np.array([[1,0],[0,1]])
    Gb = np.array([[1]])
    xc = np.array([[0],[0]])
    G = np.array([[100, 0], [0, 100]])

    GL = []
    XL = []

    for i in range(0, 3):
        G_y = C[i] @ G @ C[i].T + Gb
        K = G @ C[i].T @ np.linalg.inv(G_y)
        yt = y[i] - C[i] @ xc
        xc = xc + K @ yt
        beta = y - C[i] @ xc
        G = G - K @ C[i] @ G

        xc = A[i]@xc + u[i]
        G = A[i]@G@A[i].T + Ga
        print("x_{} = {}".format(i+1,xc))
        print("G_{} = {}".format(i+1,G))
        GL.append(G)
        XL.append(xc)

        eps([G], 0.99, xc, 'G_{}'.format(i+1))

def Q4_12():
    U = np.array([4, 10, 10, 13, 15])
    Om = np.array([5, 10, 11, 14, 17])
    Tr = np.array([0,1,5,5,3])
    C = []
    for i in range(4):
        C[i]= np.array([U[i], Tr[i]])
    A = np.identity(2)

        




import numpy as np
from numpy import cos, sin, pi
from scipy.linalg import det, expm, inv, sqrtm
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.random import randn
from celluloid import Camera

def loadcsv(file1):
    fichier = open(file1,'r')
    D = fichier.read().split("\n")
    fichier.close()
    for i in range(len(D)):
        D[i] = D[i].split(";")
    D = np.array([[float(elt) for elt in Ligne] for Ligne in D])
    return D

def kalman_correction(xc, y, C, G_x, G_b):
    yt = y - C @ xc
    G_y = C @ G_x @ C.T + G_b
    K = G_x @ C.T @ np.linalg.inv(G_y)
    xc = xc + K @ yt
    G_x = G_x - K @ C @ G_x
    return xc, G_x

def kalman_prediction(xc, G_x, A, u, G_alpha):
    xc = A@xc + u
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

def R(phi, theta, psi):
    A = np.array([[np.cos(psi), - np.sin(psi), 0],
                  [np.sin(psi), np.cos(psi), 0],
                  [0,0,1]])
    B = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0,1,0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    C = np.array([[1,0,0],
                  [0, np.cos(phi), -np.sin(phi)],
                  [0, np.sin(phi), np.cos(phi)]])
    return A@B@C

def Q_1(dt = 1e-1):
    D = loadcsv("slam_data.csv")
    p = np.array([[0],[0],[0]])
    p_point = np.array([[0],[0],[0]])
    x = np.zeros(D[:,0].shape[0]+1)
    y = np.zeros(D[:,0].shape[0]+1)
    z = np.zeros(D[:,0].shape[0]+1)
    for i in range(D[:,0].shape[0]):
        p_point = R(D[i,1], D[i,2], D[i,3])@(np.array([D[i,4],D[i,5],D[i,6]]).reshape((3,1)))
        p = p + dt*p_point
        x[i+1] = p[0,0]
        y[i+1] = p[1, 0]
        z[i+1] = p[2, 0]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("Estimation of the robot's path using Euler's method with dt = {}".format(dt))
    plt.show()

def g(k):
    D = loadcsv("slam_data.csv")
    phi = D[:,1]
    theta = D[:,2]
    psi = D[:,3]
    v = D[:, 4:7]
    depth = D[:, 7]
    alt = D[:,8]
    T = np.array([[1054, 1092, 1374, 1748, 3038, 3688, 4024, 4817, 5172, 5232, 5279, 5688],
                  [1, 2, 1, 0, 1, 5, 4, 3, 3, 4, 5, 1],
                  [52.4, 12.47, 54.4, 52.7, 27.73, 27.0, 37.9, 36.7, 37.37, 31.03, 33.5, 15.05]])
    t = T[0,:]
    i = T[1,:]
    r = T[2,:]
    for n in range(t.shape[0]):
        if k == t[n]:
            C = np.zeros((3, 15))
            C[0,0]=1
            C[1,1]=1
            C[2,2]=1
            C[0, 3+n-1]=-1
            C[1, 3 + n] = -1
            G_beta = 0.01*np.identity(3)
            Rk = R(phi[k], theta[k], psi[k])
            e = Rk@np.array([[0], [-np.sqrt(r[n]**2 - alt[k]**2)], [-alt[k]]])
            y = np.array([[e[0,0]], [e[1,0]], [depth[k]]])
            return y, C, G_beta
    C = np.zeros((1,15))
    C[0,2]=1
    G_beta = 0.01*np.identity(1)
    y = np.array([[alt[k]]])
    return y, C, G_beta

def Q5():
    D = loadcsv("slam_data.csv")
    dt = 0.1
    phi = D[:, 1]
    theta = D[:, 2]
    psi = D[:, 3]
    v = D[:, 4:7]
    depth = D[:, 7]
    alt = D[:, 8]
    #état initial nul
    xc = np.zeros((15,1))
    #pas d'incertitude sur la position mais une grosse sur es amers puisqu'ils ont vraisemblablement une position non nulle
    Gx = 10000*np.identity(15)
    Gx[0,0]=0
    Gx[1,1]=0
    Gx[2, 2] = 0
    # Pas de bruit sur la position des amers (immobiles) mais il y en un de std = 0.01 sur la position
    G_alpha = np.zeros((15,15))
    G_alpha[0, 0] = 0.01
    G_alpha[1, 1] = 0.01
    G_alpha[2, 2] = 0.01
    #La matrice A est l'identité
    A = np.identity(15)
    for k in range(1, D[:,0].shape[0]):
        uk = dt*np.concatenate(((R(phi[k], theta[k], psi[k])@(v[k].reshape((3,1)))), np.zeros((12,1))), axis = 0)
        y, C, G_beta = g(k)
        xc, Gx = kalman_correction(xc, y, C, Gx, G_beta)
        xc, Gx = kalman_prediction(xc, Gx, A, uk, G_alpha)
        print(xc, Gx)



















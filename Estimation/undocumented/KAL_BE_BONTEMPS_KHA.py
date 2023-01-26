import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from scipy.linalg import det, expm, inv, sqrtm
from numpy.random import randn

def loadcsv(file1):
    fichier = open(file1, 'r')
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

def kalman_prediction(xc, G_x, A, u, alpha, G_alpha):
    xc = A@xc + u + alpha
    G_x = A@G_x@A.T + G_alpha
    return xc, G_x

def eps(G, eta, mean=0, color = 'blue'):
    a = np.sqrt(-2 * np.log(1 - eta))
    theta = np.linspace(0, 2 * np.pi, 100)
    for i in range(len(G)):
        X, Y = [], []
        for th in theta:
            x, y = mean + a * sqrtm(G[i]) @ np.array([[np.cos(th)], [np.sin(th)]])
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, color = color)

DATA = loadcsv("slam_data.csv")

def rotate(p):
    phi, theta, psi = p
    A = np.array([[cos(psi), - sin(psi), 0],
                  [sin(psi), cos(psi), 0],
                  [0, 0, 1]], dtype='float')
    B = np.array([[cos(theta), 0, sin(theta)],
                  [0, 1, 0],
                  [-sin(theta), 0, cos(theta)]], dtype='float')
    C = np.array([[1, 0, 0],
                  [0, cos(phi), -sin(phi)],
                  [0, sin(phi), cos(phi)]], dtype='float')
    return A @ B @ C

def euler(data, f=rotate):
    X = np.zeros((3, len(data)))
    dt = data[1][0] - data[0][0]
    for i in range(len(data)):
        pos = data[i][1:4]
        vel = data[i][4:7]
        X[:, i] = X[:, i - 1] + dt * f(pos) @ vel
    return X

def Q1():
    X = euler(DATA)
    ax = plt.axes(projection='3d')
    x, y, z = [], [], []
    for i in range(np.shape(X)[1]):
        a, b, c = X[:, i]
        x.append(a)
        y.append(b)
        z.append(c)

    ax.plot3D(x, y, z)
    plt.show()

"""
Using the Kalman filter as predictor, calculate the precision with which the robot knows its position at each
moment t = k · dt. Give, in function of t, the standard deviation of the error over the position. What will this become
after an hour ? After two hours ? Verify your calculations experimentally by implementing a Kalman predictor
"""

def Q3():
    dt, sig = 0.1, 1
    I = np.eye(3)

    p_hat = np.zeros((3, 1))
    A = I
    cov_p = 0 * I
    cov_alpha = dt ** 2 * sig ** 2 * I
    alpha = cov_alpha @ randn(len(p_hat), 1)

    sd = [] # standard deviation

    for i in range(1, len(DATA)):
        v_bar = DATA[i][4:7].reshape(3, 1)
        u = dt * rotate(p_hat) @ v_bar
        p_hat, cov_p = kalman_prediction(p_hat, cov_p, A, u, alpha, cov_alpha)

        sd.append(np.sqrt(np.diag(cov_p)))

    def standard_deviation(t, tab=sd):
        return tab[int(t / 0.1)]

    print('1 heure: ', standard_deviation(3600))


def g(k):
    t, phi, theta, psi, vx, vy, vz, pz, a = DATA[k]
    y = pz # depth of the robot

    C = np.zeros((1, 15))
    C[0, 2] = 1
    cov_beta = 0.01

    T = np.array([[1054, 1092, 1374, 1748, 3038, 3688, 4024, 4817, 5172, 5232, 5279, 5688],
                 [1, 2, 1, 0, 1, 5, 4, 3, 3, 4, 5, 1],
                 [52.4, 12.47, 54.4, 52.7, 27.73, 27.0, 37.9, 36.7, 37.37, 31.03, 33.5, 15.05]])

    j = list(np.where(T[0, :] == t)[0])

    if j:
        j = int(j[0])
        r = T[2, j]
        Rk = rotate([phi, theta, psi])
        e = Rk @ np.array([[0], [-np.sqrt(r**2 - a**2)], [-a]])
        y = np.block([[e[0:2]], [y]])
        C = np.block([[np.zeros((2, 15))], [C]])
        jm = 3 + 2 * int(T[1, j])
        C[0, 0], C[0, jm] = 1, -1
        C[1, 1], C[1, jm + 1] = 1, -1
        cov_beta = np.diag([1, 1, 0.01])

    return y, C, cov_beta

def Q5():
    T = np.array([[1054, 1092, 1374, 1748, 3038, 3688, 4024, 4817, 5172, 5232, 5279, 5688],
                  [1, 2, 1, 0, 1, 5, 4, 3, 3, 4, 5, 1],
                  [52.4, 12.47, 54.4, 52.7, 27.73, 27.0, 37.9, 36.7, 37.37, 31.03, 33.5, 15.05]])
    t = T[0]
    ai = T[1]
    r = T[2]
    dt = 0.1
    x_hat = np.zeros((15, 1))
    cov_x = np.diag(3 * [0] + 12 * [10**5])

    cov_alpha = np.diag(3 * [0.01] + 12 * [0])
    alpha = 0
    A = np.eye(15)
    plt.figure()
    plt.xlim([-200, 1000])
    plt.ylim([-400, 800])
    for i in range(len(DATA)):
        v_bar = DATA[i][4:7].reshape(3, 1)
        u = np.block([[dt * rotate(DATA[i][1:4]) @ v_bar], [np.zeros((12, 1))]])
        y, C, cov_beta = g(i)
        x_hat, cov_x = kalman_correction(x_hat, y, C, cov_x, cov_beta)
        x_hat, cov_x = kalman_prediction(x_hat, cov_x, A, u, alpha, cov_alpha)
        if (i%200==0):
            eps([cov_x[0:2, 0:2]], 0.9, mean = x_hat[0:2], color = 'blue')
            for n in range(3, len(x_hat)-1, 2):
                eps([cov_x[n:n+2, n:n+2]], 0.9, mean=x_hat[n:n+2], color = 'red')
    return x_hat, cov_x


if __name__ == '__main__':
    x_hat, cov_x = Q5()
    for i in range(3, len(x_hat)-1, 2):
        print(f'Position amer {int(1+(i-3)/2)}: ({x_hat[i][0]}, {x_hat[i+1][0]})')
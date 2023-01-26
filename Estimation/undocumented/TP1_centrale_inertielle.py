import numpy as np
import scipy.linalg as sc
from math import *
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, expm, norm, block_diag

M = np.array([[0, 0, 10, 0, 0, 10, 0, 0],
                  [-1, 1, 0, -1, -0.2, 0, 0.2, 1],
                  [0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1]])

def Q1() :

    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot3D(M[0], M[1], M[2]+2)
    ax.plot3D(M[0], M[1], 0*M[2], color = 'black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-10,10])
    ax.set_ylim3d([-10, 10])
    ax.set_zlim3d([-5, 5])
    ax.set_title("représentation de l'AUV")
    plt.show()

def f(x,u):
    return np.array([x[3]*cos(x[5])*cos(x[6]),
                     x[3]*cos(x[5])*sin(x[6]),
                     -x[3]*sin(x[5]),
                     u[0],
                     -0.1*sin(x[6])*cos(x[5]) + tan(x[5])*x[3]*(sin(x[4])*u[1]+cos(x[4])*u[2]),
                    cos(x[4])*x[3]*u[1]-sin(x[3])*x[2]*u[2],
                    (sin(x[4])/cos(x[5]))*x[3]*u[1]+ (cos(x[4])/cos(x[5]))*x[3]*u[2]])

def rotation_H(X):
    px, py, pz, v, phi, theta, psi = X[0], X[1], X[2], X[3], X[4], X[5], X[6]
    b = np.array([px, py, pz]).reshape((3,1))
    Ad_i = np.array([[0,0,0], [0,0,-1], [0,1,0]])
    Ad_j = np.array([[0,0,1], [0,0,0], [-1,0,0]])
    Ad_k = np.array([[0,-1,0],[1,0,0], [0,0,0]])
    E = sc.expm(psi*Ad_k) @ sc.expm(theta*Ad_j)@sc.expm(psi*Ad_i)
    R = np.concatenate((E,b), axis = 1)
    R = np.concatenate((R, np.array([[0,0,0,1]])), axis = 0)
    H = np.dot(R, M)
    return H

def vect_rot(X, u):
    px, py, pz, v, phi, theta, psi = X[0], X[1], X[2], X[3], X[4], X[5], X[6]
    phi_p = f(X,u)[4]
    theta_p = f(X,u)[5]
    psi_p = f(X, u)[6]
    Ad_i = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    Ad_j = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    Ad_k = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    E = sc.expm(psi * Ad_k) @ sc.expm(theta * Ad_j) @ sc.expm(psi * Ad_i)
    v = np.linalg.inv(E)@np.array([[psi_p, theta_p, phi_p]]).reshape((3,1))
    return v

def simul(X,u, dt, t_end):
    t=0
    while t<t_end:
        X = X +dt*f(X,u)
        t+=dt
    return X


def Q2():
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    X = np.array([0,0,1,0,0,0,np.pi/3])
    H = rotation_H(X)
    ax.plot3D(H[0], H[1], H[2])
    ax.plot3D(H[0], H[1], 0*H[2], color = 'black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-10, 10])
    ax.set_ylim3d([-10, 10])
    ax.set_zlim3d([-10, 10])
    ax.set_title("représentation de l'AUV avec rotation d'angle psi = 60°")
    plt.show()

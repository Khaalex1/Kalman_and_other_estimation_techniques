import numpy as np
import scipy.linalg as sc
from math import *
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, expm, norm, block_diag


"""Exercises from ensi_isterobV2 ~p.139"""

"""Ex 6.1"""

#Q1

def f(x,y):
    return x*y


def ls(A,Y):
    """
    Least squares estimator. Finds X approximating AX = Y with LS estimation
    :param A: numpy array (m,n)
    :param Y: numpy array (m,p)
    :return: estimated solution numpy array (n,p)
    """
    ATA = A.T @ A
    B = A.T @ Y
    return np.linalg.solve(ATA,B)


def ex1_Q2() :
    X1 = np.linspace(-5,5,10)
    Y1 = np.linspace(-5,5,10)

    x1,y1 = np.meshgrid(X1,Y1)
    u1 = y1
    v1 = x1

    plt.figure()
    plt.grid()
    # strm = plt.streamplot(x1,y1,u1,v1, color = np.sqrt(u1**2 + v1**2),linewidth=2,cmap = plt.cm.viridis)
    # plt.colorbar(strm.lines)
    plt.quiver(x1, y1, u1, v1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("champ de vecteurs du gradient de f")
    plt.show()

def ex1_Q3() :
    X1 = np.linspace(-10, 10, 100)
    Y1 = np.linspace(-10, 10, 100)
    Xm, Ym = np.meshgrid(X1,Y1)
    Zm = f(Xm, Ym)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour(Xm, Ym, Zm, 10, offset = -1)
    ax.plot_surface(Xm, Ym, Zm)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-10, 10])
    ax.set_ylim3d([-10, 10])
    ax.set_zlim3d([-100, 100])
    ax.set_title("graphe de f(x,y) = xy")
    plt.show()

# Q4

def g(x,y):
    return 2*x**2 + x*y + 4*y**2 + y - x +3

def ex1_Q4() :
    X1 = np.linspace(-10, 10, 100)
    Y1 = np.linspace(-10, 10, 100)

    x1, y1 = np.meshgrid(X1, Y1)
    u1 = 4*x1 + y1 -1
    v1 = x1 + 8*y1 +1

    plt.figure()
    plt.grid()
    strm = plt.streamplot(x1, y1, u1, v1, color=np.sqrt(u1 ** 2 + v1 ** 2), linewidth=2, cmap=plt.cm.viridis)
    plt.colorbar(strm.lines)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("champ de vecteurs du gradient de g représentés selon leur norme")
    plt.show()

    X1 = np.linspace(-10, 10, 100)
    Y1 = np.linspace(-10, 10, 100)
    Xm, Ym = np.meshgrid(X1, Y1)
    Zm = g(Xm, Ym)


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour(Xm, Ym, Zm, 10, offset=-1)
    ax.plot_surface(Xm, Ym, Zm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-10, 10])
    ax.set_ylim3d([-10, 10])
    ax.set_zlim3d([-100, 500])
    ax.set_title("graphe de g(x,y)")
    plt.show()

def ex2_Q1():
    T = [-3, -1, 0, 2, 3, 6]
    Y = np.array([17,3,1,5,11,46])
    A = np.zeros((6,3))
    for i in range(6):
        A[i] = np.array([T[i]**2, T[i], 1])
    ATA = A.T @ A
    B = A.T @ Y
    return np.linalg.solve(ATA, B)

#p1 = 1.41551565
#p2  = -0.98476059
#p3 = 1.06298343

def ex2_Q2():
    T = [-3, -1, 0, 2, 3, 6]
    Y = np.array([17, 3, 1, 5, 11, 46])
    A = np.zeros((6, 3))
    for i in range(6):
        A[i] = np.array([T[i] ** 2, T[i], 1])
    ATA = A.T @ A
    B = A.T @ Y
    p_c = np.linalg.solve(ATA, B)
    Y_c = A@p_c

    plt.figure()
    plt.grid()
    plt.plot(T, Y_c, label = "mesures filtrées Yc", color = 'red')
    plt.scatter(T, Y, label="vraies mesures Y", color='blue')
    plt.plot(T, np.abs(Y - Y_c ), label = 'résidu Y - Yc', color = 'green')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Y')
    plt.title('Y, Y filtré et résidu en fonction du temps')
    plt.show()
    print('mesures filtrées Yc = ', Y_c)
    print('résidu Y - Yc = ', Y - Y_c )

def ex3_Q1():
    Om = np.array([5, 10, 8, 14, 17])
    U = np.array([[4, 10, 10, 13, 15]]).reshape((5, 1))
    Tr = np.array([[0, 1, 5, 5, 3]]).reshape((5, 1))
    A = np.concatenate((U,Tr), axis = 1)
    ATA = A.T @ A
    B = A.T @ Om
    p_c = np.linalg.solve(ATA, B)
    Y_c = A @ p_c
    print('p1, p2 = {}'.format(p_c))
    print('résidu = {}'.format(Om - Y_c))

#p1 = 1.18831169
#p2 = -0.51688312

def ex3_Q2():
    Om = np.array([5, 10, 8, 14, 17])
    U = np.array([[4, 10, 10, 13, 15]]).reshape((5, 1))
    Tr = np.array([[0, 1, 5, 5, 3]]).reshape((5, 1))
    A = np.concatenate((U, Tr), axis=1)
    ATA = A.T @ A
    B = A.T @ Om
    p_c = np.linalg.solve(ATA, B)
    Y_c = A @ p_c
    p1, p2 = p_c
    U = 20
    Tr = 10
    print('Omega(U=20, T=10) = {} rad/s'.format(int((p1*U + p2*Tr)*100)/100))


if __name__ == "__main__":
    ex1_Q2()


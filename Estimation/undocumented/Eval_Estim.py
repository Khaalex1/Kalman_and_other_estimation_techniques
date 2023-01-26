import numpy as np
import scipy.linalg as sc
from math import *
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, expm, norm, block_diag
import random

def Q1():
    x = np.array([2,2.2, 2.4, 2.6, 2.8, 3])
    y = np.array([20, 28, 38, 44, 59, 76])
    a_r = []
    b_r=[]
    a_b=[]
    b_b=[]
    Al = np.zeros((2000, 2))
    for i in range(2000):
        Al[i]= np.array([random.uniform(0,4), random.uniform(2,3)])
    for i in range(2000):
        count =0
        for k in range(6):
            if np.abs(y[k]-Al[i,0]*x[k]**Al[i,1])<10:
                count +=1
        if count ==6:
            a_r.append(Al[i,0])
            b_r.append(Al[i,1])
        else :
            a_b.append(Al[i,0])
            b_b.append(Al[i,1])
    plt.figure()
    plt.grid()
    plt.scatter(a_r, b_r, color = 'red')
    plt.scatter(a_b, b_b, color='blue')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title('points (a,b) acceptés par Monté-Carlo')

def Q2():
    x = np.array([[2, 2.2, 2.4, 2.6, 2.8, 3]]).T
    y = np.array([[20, 28, 38, 44, 59, 76]]).T
    y2 = np.log(y)
    x2 = np.log(x)
    M = np.concatenate((x2, np.ones((6,1))), axis = 1)
    MTM = M.T @ M
    B = M.T @ y2
    Xc = np.linalg.solve(MTM, B)
    a = np.exp(Xc[1,0])
    b = Xc[0,0]
    ecart = y - a*x**b
    print('a = {}, b = {}, y - yc = {}'.format(a, b, ecart))
    plt.figure()
    plt.grid()
    plt.scatter(x, a*x**b, label = 'estimation de y', color = 'blue' )
    plt.scatter(x, y, label='y réel', color='red')
    plt.scatter(x, ecart, label='y - yc', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y vs yc')



A = np.array([[0,5,5,0],
              [0,0,5,5]])
B = np.array([[5,5,0,0],
              [0,5,5,0]])
Y = np.array([4, 1.1, 1, 1.9])

#liste des 8 distances enregistrées par le lidar
def f(p):
    D = []
    m = np.array([p[0], p[1]])
    #balayage angulaire : pour chaque angle
    for i in range(4):
        u = np.array([[np.cos(p[2]+ i*np.pi/2),np.sin(p[2]+ i*np.pi/2)]])
        #valeur de référence très grande pour pouvoir prendre un min  sur les distances
        ref = 1000
        #Pour chaque colonne de A et B (murs)
        for j in range(A.shape[1]):
            #np.array( [x , y] ) -> np.array( [ [x] , [y] ] ) -> np.array( [ [x , y] ] )
            a = A[:,j].reshape((2,1)).T
            b = B[:,j].reshape((2,1)).T
            #Si vérification d'obstacle avec déterminant, on enregistre d pourvu que ce soit la plus faible des distances si 2 murs sont traversés
            if np.linalg.det(np.concatenate((a-m, u), axis = 0))*np.linalg.det(np.concatenate((b-m, u), axis = 0))<=0 and np.linalg.det(np.concatenate((a-m, b-a), axis = 0))*np.linalg.det(np.concatenate((u, b-a), axis = 0))>=0:
                d = np.linalg.det(np.concatenate((a-m, b-a), axis = 0))/np.linalg.det(np.concatenate((u, b-a), axis = 0))
                d = min(ref, d)
                ref = d
        D.append(ref)
    return D


def draw_room():
    #plt.grid()
    for j in range(A.shape[1]):
        #plt.plot([xA, xB], [yA, yB])
        plt.plot(np.array([A[0,j], B[0, j]]), np.array([A[1,j], B[1,j]]), color = 'grey')

def update_rob(p,y, clr ='red', bool=True):
    my_plot, = plt.plot([], [], color = clr)
    X=  np.zeros((4,2))
    Y = np.zeros((4,2))
    for i in range(4):
        X[i] = p[0] + np.array([0, y[i] * np.cos(p[2] + i * np.pi / 2)])
        Y[i] = p[1] + np.array([0, y[i] * np.sin(p[2] + i * np.pi / 2)])
    my_plot.set_data(X, Y)  # ligne qui permet l'animation de la courbe
    plt.pause(1e-6)  # ceci est necessaire pour afficher l'animation
    if bool:
        my_plot.remove()


def recuit():
    T= 5
    p0 = np.array([random.uniform(0,5),random.uniform(0,5),2*np.pi*random.uniform(0,1)])
    yc = f(p0)
    draw_room()
    while T>1e-2:
        x = random.uniform(-1  ,1)
        y= random.uniform(-1,1)
        theta =  random.uniform(0, 2*np.pi)
        p1 = p0 + T*np.array([x,y,theta])
        j0 = np.linalg.norm(f(p0) - Y)
        j1 = np.linalg.norm(f(p1) - Y)
        if j1<j0:
            p0 = p1
            print("Température = {}, p = {}, yc = {}, norm = {}, ".format(T, p0, f(p0), np.linalg.norm(f(p0)-Y)))
            update_rob(p1, f(p1), 'green')
        T = 0.99*T
        update_rob(p1, f(p1))
    update_rob(p0,f(p0), 'blue', False)
    print("p0 = {}, yc = {}, ||y - yc|| = {}".format(p0, f(p0), np.linalg.norm(Y-f(p0))))








import numpy as np
import scipy.linalg as sc
from math import *
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, expm, norm, block_diag
import random

def ex3_5():
    #2000 points aléatoires (a,b) stockés dans P
    P = np.zeros((2000, 2))
    a_b = []
    b_b=[]
    a_r = []
    b_r = []
    for i in range(2000):
        P[i]= np.array([random.uniform(0,2), random.uniform(0,2)])
    y_true = np.array([0,1,2.5, 4.1, 5.8, 7.5])
    C = np.array([1,1])
    plt.figure()
    plt.grid()
    for i in range(2000):
        count = 0
        X = np.array([[0], [0]])
        #Pour les 2000 poinrs aléatoires, on vérifie que X1, ..., X5 remplissent bien le critère (<0.5) pour les stocker dans les listes rouges
        if np.abs(y_true[0] - C @ X) < 0.5:
            count += 1
        for k in range(1,5):
            X = np.array([[1, 0],[P[i,0], 0.3]])@X + np.array([[P[i,1]], [1-P[i,1]]]).reshape((2,1))
            if np.abs(y_true[k]-C@X)<0.5:
                count +=1
        if count ==5:
            a_r.append(P[i,0])
            b_r.append(P[i,1])
        #les autres points sont stockés dans les listes bleues
        else :
            a_b.append(P[i, 0])
            b_b.append(P[i, 1])
    plt.scatter(a_b, b_b, color = 'blue', label = 'refusé')
    plt.scatter(a_r, b_r, color = 'red', label = "accepté")
    plt.xlabel('a')
    plt.ylabel('b')
    plt.legend()
    plt.title('Points (a,b) acceptables par Monté Carlo')
    plt.show()

A = np.array([[0,7,7,9,9,7,7,4,2,0,5,6,6,5],
                  [0,0,2,2,4,4,7,7,5,5,2,2,3,3]])
B = np.array([[7,7,9,9,7,7,4,2,0,0,6,6,5,5],
                  [0,2,2,4,4,7,7,5,5,0,2,3,3,2]])
Y = np.array([6.4, 3.6, 2.3, 2.1, 1.7, 1.6, 3.0, 3.1])

#liste des 8 distances enregistrées par le lidar
def f(p):
    D = []
    m = np.array([p[0], p[1]])
    #balayage angulaire : pour chaque angle
    for i in range(8):
        u = np.array([[np.cos(p[2]+ i*np.pi/4),np.sin(p[2]+ i*np.pi/4)]])
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
    plt.grid()
    for j in range(A.shape[1]):
        #plt.plot([xA, xB], [yA, yB])
        plt.plot(np.array([A[0,j], B[0, j]]), np.array([A[1,j], B[1,j]]), color = 'grey')

#y = f(p) (la liste des 8 distances)
def draw_rob(p, y, clr='red'):
    #pour chaque angle
    for i in range(8):
        #plt.plot([x, x+d*cos(theta)], [y, y + d*sin(theta)])
        plt.plot(p[0]+ np.array([0, y[i]*np.cos(p[2]+ i*np.pi/4)]), p[1] + np.array([0, y[i]*np.sin(p[2] + i*np.pi/4)]), color = clr)

def update_rob(p,y, clr ='red'):
    my_plot, = plt.plot([], [], color = clr)
    X=  np.zeros((8,2))
    Y = np.zeros((8,2))
    for i in range(8):
        X[i] = p[0] + np.array([0, y[i] * np.cos(p[2] + i * np.pi / 4)])
        Y[i] = p[1] + np.array([0, y[i] * np.sin(p[2] + i * np.pi / 4)])
    my_plot.set_data(X, Y)  # ligne qui permet l'animation de la courbe
    plt.pause(1e-2)  # ceci est necessaire pour afficher l'animation
    my_plot.remove()

def test():
    p = np.array([3,4,np.pi/8])
    draw_room()
    draw_rob(p, f(p))

def recuit():
    T= 5
    p0 = np.array([random.uniform(0,4),random.uniform(0,4),random.uniform(0,1)])
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
        T = 0.965*T
        update_rob(p1, f(p1))
    draw_rob(p0,f(p0), 'blue')
    print("p0 = {}, yc = {}, ||y - yc|| = {}".format(p0, f(p0), np.linalg.norm(Y-f(p0))))

recuit()






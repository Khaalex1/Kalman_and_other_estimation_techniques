import numpy as np
from numpy import cos, sin, pi
from scipy.linalg import det, expm, inv, sqrtm
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.random import randn
from celluloid import Camera

def eps(G, eta, mean=0):
    a = np.sqrt(-2 * np.log(1 - eta))
    theta = np.linspace(0, 2 * np.pi, 100)
    for i in range(len(G)):
        X, Y = [], []
        for th in theta:
            x, y = mean + a * sqrtm(G[i]) @ np.array([[np.cos(th)], [np.sin(th)]])
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, color = "pink")


def kalman_correction(xc, y, C, G_x, G_b=np.array([[0]])):
    yt = y - C @ xc
    G_y = C @ G_x @ C.T + G_b
    K = G_x @ C.T @ np.linalg.inv(G_y)
    xc = xc + K @ yt
    G_x = G_x - K @ C @ G_x
    return xc, G_x


def kalman_prediction(xc, G_x, A, u, G_alpha=0):
    xc = A @ xc + u
    G_x = A @ G_x @ A.T + G_alpha
    return xc, G_x

def kalman(xc, G_x, G_b, A, u, G_alpha,  y=np.array([]), C = np.array([])):
    if y.size==0 or C.size ==0:
        xc, G_x = kalman_correction(xc, y, C, G_x, G_b)
    else :
        x_c , G_x = kalman_correction(xc, y, C, G_x, G_b)
        x_c, G_x = kalman_prediction(xc, G_x, A, u, G_alpha)
    return x_c, G_x

#corection
def Q_4_13():
    a = [np.array([[0],[0]]),np.array([[2],[1]]),np.array([[15],[5]]),np.array([[3],[12]])]
    b = [np.array([[0],[0]]),np.array([[15],[5]]),np.array([[3],[12]]),np.array([[2],[1]])]
    d = [0,2,5,4]
    u = [np.array([[0],[0]]),np.array([[0],[0]]),np.array([[0],[0]]),np.array([[0],[0]])]
    for i in range(1,4):
        u[i]= (b[i]-a[i])/np.linalg.norm(b[i]-a[i])
    xc = np.array([[1],
                   [2]])
    G_x = np.array([[100, 0],
                    [0, 100]])
    G_b = np.array([[1]])
    for i in range(1,4):
        y = d[i] + u[i][0,0]*a[i][1,0]-u[i][1,0]*a[i][0,0]
        C = np.array([[-u[i][1,0], u[i][0,0]]])

        xc, G_x = kalman_correction(xc, y, C, G_x, G_b)

        print("y[{}] = \n {} ".format(i, y))
        print("C[{}] = \n {} ".format(i, C))
    print("xc[{}] = \n {} ".format(i, xc))
    print("G_x[{}] = \n {} ".format(i,G_x))


# Q 4.17 1)

# def euler(t0, tf, y0, h, f):
#     T = np.arange(t0, tf, h)
#     Y = np.zeros((len(y0), len(T)))
#     Y[:,0] = y0
#     for i in range(1, len(T)):
#         Y[:,i]= Y[:, i-1] + h*f(T[i-1], Y[:, i-1])
#     return T,Y

def f1(x, u):
    return np.array([x[3,0]*np.cos(x[4,0])*np.cos(x[2,0]), x[3,0]*np.cos(x[4,0])*np.sin(x[2,0]), x[3,0]*np.sin(x[4,0])/3 , u[0,0], u[1,0]])

def simul1():
    x = np.array([[0,0,np.pi/3,4,0.3]]).T
    dt = 0.1
    T = np.arange(0,10, dt)
    G_alpha = dt*np.diag([0,0,0.01, 0.01, 0.01])
    camera = Camera(plt.figure())
    for i in range(0,len(T)):
        if i == 0:
            plt.scatter(x[0,0], x[1,0], c = 'red', s = 150)
            camera.snap()
        else :
            alpha = np.array([[0],[0], [np.random.randint(1)], [np.random.randint(1)], [np.random.randint(1)]])
            x = x + f1(x, np.array([[0], [0]])) * dt + alpha
            plt.scatter(x[0,0], x[1,0], c = 'red', s = 150)
            camera.snap()
    anim = camera.animate(blit = True)
    anim.save('s1.mp4')


# Q 4.17 2)
def f2(Z, delta, theta, u):
    C = np.array([[0,0,np.cos(delta)*np.cos(theta)], [0,0, np.cos(delta)*np.sin(theta)],[0,0,0]])
    return C@Z + np.array([0,0,u[0]])

# Q 4.17 3)

def simul2():
    dt = 0.1
    T = np.arange(0, 40, dt)
    z = np.array([[0,0,4]]).T
    G_z = 0*np.identity(3)
    uz = np.array([[0], [0], [0]])
    G_alpha_z = dt*np.diag([0.01,0.01,0.01])
    camera = Camera(plt.figure())
    theta = (np.pi/4)*np.ones(len(T))
    sigma= (np.pi/4)*np.ones(len(T))

    for i in range(len(T)):
        if i==0:
            plt.scatter(z[0, 0], z[1, 0], c='red', s=50, label = "robot Kalman")
            eps([G_z[0:2, 0:2]], 0.9, z[0:2])
            camera.snap()
        else:
            A = np.array([[1, 0, dt*np.cos(i*np.pi/180)*np.cos(3*i*np.pi/180)], [0,1, dt*np.cos(i*np.pi/180)*np.sin(3*i*np.pi/180)],[0,0,1]])
            #y = x[3,0]
            #C = np.array([[0,0,1]])
            #G_b = 0.1
            #z, G_z = kalman(z, G_z, G_b, A, uz, G_alpha_z, y=y, C=C)
            #z, G_z = kalman_correction(z, y, C, G_z, G_b)
            z, G_z  = kalman_prediction(z, G_z, A, uz, G_alpha_z )
            plt.scatter(z[0, 0], z[1, 0], c='red', s=  50)
            eps([G_z[0:2,0:2]], 0.9, z[0:2])
            camera.snap()
    anim = camera.animate(blit=True)
    anim.save("s1.gif")


# Exo 4.1.8
def gonio():
    Amers = [np.array([[0], [25]]), np.array([[15], [30]]), np.array([[30], [15]]), np.array([[15], [20]])]
    amerx = []
    amery = []
    for i in range(4):
        amerx.append(Amers[i][0,0])
        amery.append(Amers[i][1,0])

    dt = 0.1
    T = np.arange(0, 30, dt)
    z = np.array([[10, 10, 4]]).T
    G_z = np.zeros((3,3))
    uz = np.array([[0], [0], [0]])
    G_alpha_z = dt * np.diag([0.01, 0.01, 0.01])
    camera = Camera(plt.figure())
    delta = np.pi/5
    theta = 0
    for i in range(len(T)):
        if i == 0:
            plt.scatter(z[0, 0], z[1, 0], c='green', s=50)
            eps([G_z[0:2, 0:2]], 0.9, z[0:2])
            plt.scatter(amerx, amery, color="blue", s=100)
            camera.snap()
        else:
            theta = theta + dt*(z[2,0]*np.sin(delta/3))
            A = np.array([[1, 0, dt * np.cos(delta) * np.cos(theta)],
                          [0, 1, dt * np.cos(delta) * np.sin(theta)],
                          [0, 0, 1]])
            for amer in Amers:
                if np.linalg.norm(z[0:2]-amer)<15:
                    angle_ai = np.arctan((amer[1,0]-z[1,0])/(amer[0,0]-z[0,0]))
                    C = np.array([[-np.sin(angle_ai), np.cos(angle_ai), 0]])
                    y = C@z
                    z, G_z = kalman_correction(z, y, C, G_z, G_b = np.array([[1]]))
                    z, G_z = kalman_prediction(z, G_z, A, uz, G_alpha_z)
                    plt.plot([z[0, 0], amer[0, 0]], [z[1, 0], amer[1, 0]], color='red')
                else :
                    z, G_z = kalman_prediction(z, G_z, A, uz, G_alpha_z)
            plt.scatter(z[0, 0], z[1, 0], c='green', s=50)
            plt.scatter(amerx, amery, color="blue", s=100)
            eps([G_z[0:2, 0:2]], 0.9, z[0:2])
            camera.snap()
    anim = camera.animate(blit=True)
    anim.save("s2.gif")









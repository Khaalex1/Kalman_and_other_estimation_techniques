import numpy as np
from numpy import cos, sin, pi
from scipy.linalg import det, expm, inv, sqrtm
import matplotlib.pyplot as plt
from numpy.random import randn

xbar = np.array([[1], [2]])
Gx = np.array([[1,0], [0,1]])
invGx = np.linalg.inv(Gx)
n = 2

x1, x2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5,5,0.1))
X0 = x1 - xbar[0]
X1 = x2 - xbar[1]

Q = invGx[0,0]*X0**2 +  invGx[1,1]*X1**2 +  (invGx[1,0] + invGx[0,1])*X0*X1
pi_x = (1/np.sqrt((2*np.pi)**n))*np.exp(-0.5*Q)
fig = plt.figure(1)
ax = plt.axes(projection = '3d')
ax.contour(x1,x2, pi_x)

ax.plot_surface(x1,x2, pi_x)

theta = np.pi/6
A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
B = np.array([[1,0],  [0,3]])
V = np.array([[2], [-5]])
ybar = A@B@xbar + V

Y0 = x1 - ybar[0]
Y1 = x2 - ybar[1]

Gy = (A@B).T@Gx@(A@B)
invGy = np.linalg.inv(Gy)

Q = invGy[0,0]*Y0**2 +  invGy[1,1]*Y1**2 +  (invGx[1,0] + invGy[0,1])*Y0*Y1
pi_y = (1/np.sqrt((2*np.pi)**n))*np.exp(-0.5*Q)

fig = plt.figure(2)
ax = plt.axes(projection = '3d')
ax.contour(x1,x2, pi_y)

ax.plot_surface(x1,x2, pi_y)

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
    plt.figure()
    for i in range(len(G)):
        X, Y = [], []
        for th in theta:
            x, y = mean + a * sqrtm(G[i]) @ np.array([[np.cos(th)], [np.sin(th)]])
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, label = 'G{}'.format(i+1))
    plt.axis('square')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ellipses de confiance à {}'.format(eta))


eps(G, 0.9)

""" Exo 4.3 """

y = randn(2, 1000)
mean = np.array([[1], [2]])
gamma = np.array([[4, 3], [3, 3]])

x = mean + sqrtm(gamma) @ y
plt.figure()
plt.scatter(y[0], y[1], color='blue')
plt.scatter(x[0], x[1], color='red')

for eta in [0.9, 0.99, 0.999]:
    eps([gamma], eta, mean)
plt.axis('square')
plt.grid()
plt.show()

#Q5)

A = np.array([[0,1],[-1,0]])
B = np.array([[2],[3]])
dt = 0.01
Ad = np.identity(2) + dt*A
plt.figure()
for t in np.arange(0,1,dt):
    ud = dt*np.sin(t)*B
    x = Ad@x + dt*np.sin(t)*B*np.ones((1, 1000))
    plt.scatter(x[0], x[1], color = 'blue', s =5)






import numpy as np
from numpy import cos, sin, pi
from scipy.linalg import det, expm, inv, sqrtm
import matplotlib.pyplot as plt
from numpy.random import randn

def Q1():
    y = np.array([[10.7], [-25.5], [-1]])
    C = np.array([[2, 3, 2],
                  [1, 5, -3],
                  [-3, -2, 1]])
    return np.linalg.inv(C)@y

# array([[ 3.89166667],
#        [-2.63333333],
#        [ 5.40833333]])


def Q3():
    xbar = np.array([[0], [0], [0]])
    y = np.array([[10.7], [-25.5], [-1], [27.2]])
    C = np.array([[2, 3, 2],
                  [1, 5, -3],
                  [-3, -2, 1],
                  [4, -1, 3]])
    G_b = np.array([[16, 0, 0, 0],
                    [0, 16, 0, 0],
                    [0, 0, 9, 0],
                    [0, 0, 0, 9]])
    G_x = np.array([[ 1000, 0, 0],
                    [0, 1000, 0],
                    [0, 0, 1000]])
    G_y = C @ G_x @ C.T + G_b
    K = G_x @ C.T @ np.linalg.inv(G_y)
    yt = y - C @ xbar
    xc = xbar + K @ yt
    beta = y - C @ xc
    G_eps = G_x - K @ C @ G_x
    print("x chapeau = ", xc)
    # print("beta = ", beta)
    print("gamma epsilon = ", G_eps)

# x chapeau =
# [[ 2.83514488]
#  [-2.13311453]
#  [ 5.02804099]]
# gamma epsilon =
# [[ 0.73675441 -0.55362415 -0.72834669]
#  [-0.55362415  0.90812759  0.83318794]
#  [-0.72834669  0.83318794  1.40569632]]

def Q2():
    Y = np.array([[10.7], [-25.5], [-1], [27.2]])
    C = np.array([[2, 3, 2],
                  [1, 5, -3],
                  [-3, -2, 1],
                  [4, -1, 3]])
    CTC = C.T @ C
    B = C.T @ Y
    sol = np.linalg.solve(CTC, B)
    residu = Y - C@sol
    print("solution = ", sol)
    print("résidu = ", residu)

# solution =
# [[ 2.74279913]
#  [-2.10968447]
#  [ 5.28419244]]
# résidu =
# [[ 0.97507029]
#  [-1.84179944]
#  [-2.27516401]
#  [-1.73345829]]

def Q4():
    xbar =  np.array([[ 3.88409215],
                      [-2.6298891 ],
                      [ 5.40318847]])
    y = np.array([[10.7], [-25.5], [-1], [27.2]])
    C = np.array([[2, 3, 2],
                  [1, 5, -3],
                  [-3, -2, 1],
                  [4, -1, 3]])
    G_b = np.array([[16, 0, 0, 0],
                    [0, 16, 0, 0],
                    [0, 0, 9, 0],
                    [0, 0, 0, 9]])
    G_x = np.array([[1000, 0, 0],
                    [0, 1000, 0],
                    [0, 0, 1000]])
    G_y = C @ G_x @ C.T + G_b
    K = G_x @ C.T @ np.linalg.inv(G_y)
    yt = y - C @ xbar
    xc = xbar + K @ yt
    beta = y - C @ xc
    G_eps = G_x - K @ C @ G_x
    print("x chapeau = ", xc)
    # print("beta = ", beta)
    print("gamma epsilon = ", G_eps)
    #residu = y - C @ xc
    #print("résidu y -C@xc = ", residu)

# x chapeau =
# [[ 2.83552708]
#  [-2.13315127]
#  [ 5.03061608]]
# gamma epsilon =
# [[ 0.73675441 -0.55362415 -0.72834669]
#  [-0.55362415  0.90812759  0.83318794]
#  [-0.72834669  0.83318794  1.40569632]]

def Q4b():
    xbar =  np.array([[ 3.89166667],
       [-2.63333333],
       [ 5.40833333]])
    y = np.array([[10.7], [-25.5], [-1], [27.2]])
    C = np.array([[2, 3, 2],
                  [1, 5, -3],
                  [-3, -2, 1],
                  [4, -1, 3]])
    G_b = np.array([[16, 0, 0, 0],
                    [0, 16, 0, 0],
                    [0, 0, 9, 0],
                    [0, 0, 0, 9]])
    G_x = np.array([[50, 0, 0],
                    [0, 50, 0],
                    [0, 0, 50]])
    G_y = C @ G_x @ C.T + G_b
    K = G_x @ C.T @ np.linalg.inv(G_y)
    yt = y - C @ xbar
    xc = xbar + K @ yt
    beta = y - C @ xc
    G_eps = G_x - K @ C @ G_x
    print("x chapeau = ", xc)
    # print("beta = ", beta)
    print("gamma epsilon = ", G_eps)
    #residu = y - C@xc
    #print("résidu y -C@xc = ", residu)

# x chapeau =
# [[ 2.83553082]
#  [-2.1331543 ]
#  [ 5.03061492]]
# gamma epsilon =
# [[ 0.73675441 -0.55362415 -0.72834669]
#  [-0.55362415  0.90812759  0.83318794]
#  [-0.72834669  0.83318794  1.40569632]]







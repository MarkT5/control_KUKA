import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

J = np.zeros((20, 20))

J[6:9, 0:3] = [[1,2,3], [1,2,3], [1,2,3]]
L = np.array([[1,0,0], [2,3,0],[4,5,6]])
""" QR decomposition with scipy """
import scipy.linalg as linalg
import numpy as np # same matrix A and B as in LU decomposition
A = np.array([ [2., 1., 1.], [1., 3., 2.], [1., 0., 0]])
B = np.array([4., 5., 6.])
Q, R = linalg.qr(A) # QR decomposition with qr function
y = np.dot(Q.T, B) # Let y=Q'.B using matrix multiplication
x = linalg.solve(R, y) # Solve Rx=y
print(A.shape)
print(B. shape)
print(Q.shape)
print(R.shape)
print(y.shape)
print(x)
#(3, 3)
#(3,)
#(3, 3)
#(3, 3)
#(3,)
#[  6.  15. -23.]
def func(b, x):
    return b[0] * x / (b[1] + x)


def Jacobian(f, b, x):
    eps = 1e-10
    grads = []
    for i in range(len(b)):
        t = np.zeros_like(b).astype(float)
        t[i] = t[i] + eps
        print(t)
        print()
        grad = (f(b + t, x) - f(b - t, x)) / (2 * eps)
        grads.append(grad)
    return np.column_stack(grads)


def Gauss_Newton(f, x, y, b0, tol, max_iter):
    old = new = b0
    for itr in range(max_iter):
        old = new
        J = Jacobian(f, old, x)
        #print(J)
        #print(len(J))
        dy = y - f(old, x)
        new = old + np.linalg.inv(J.T @ J) @ J.T @ dy
        if np.linalg.norm(old - new) < tol:
            break
    return new


#f1 = func


def f3(b, X):
    return b[0] - (1 - b[1]) * b[2] * X[:, 0] ** 2 - b[3]*X[:, 1] ** 2


#x1 = np.linspace(-5, 5, 50)
#x2 = np.linspace(-5, 5, 50)
#X1, X2 = np.meshgrid(x1, x2)
#X = np.column_stack([X1.ravel(), X2.ravel()])
#print("x", len(X))
#y = f3([3, 1, 4, 6], X) + np.random.normal(0, 10, size=len(X))


#b = Gauss_Newton(f3, X, y, [1, 2, 5, 4], 1e-5, 10)
#y_hat = f3(b, X)

#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter(X[:, 0], X[:, 1], y, c=y, marker='o')
#ax.plot_surface(X[:, 0].reshape(50, 50), X[:, 1].reshape(50, 50), y_hat.reshape(50, 50))

#x = np.linspace(0, 5, 50)
#y = f1([2, 3], x) + np.random.normal(0, 0.1, size=50)

#a, b = Gauss_Newton(f1, x, y, [5, 1], 1e-5, 10)

#y_hat = f1([a, b], x)

#plt.scatter(x, y)
#plt.plot(x, y_hat)
#plt.show()

import numpy as np
a = np.array([[1,2],[3,4],[5,6],[7,8]])
w = [1,2,3,4]
c = np.sum(a, axis=1)
print(c)
d = np.where(a==[5,6])[0][0]
b = np.diag(w)/c
print(d)
print(np.dot(b,a))

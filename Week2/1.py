import numpy as np

t = np.array([-1, -1, 0, 2, 0, -2, 0, 0, 1, 1]).reshape(2, -1)
print(t)
c = np.matmul(t, np.transpose(t)) / t.shape[1]
print(c)
a, b = np.linalg.eig(c)
print(a)
print(b)
b = np.transpose(b)
print(b)
c = b.dot(c).dot(np.transpose(b))
print(c)
y = b[0, :].reshape(-1, 2).dot(t)
print(y)

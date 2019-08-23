import numpy as np
from PIL import Image
from scipy import signal
import scipy.misc
import matplotlib.pyplot as plt

img = 'F:/Code/2019 Summer/Week2/frog2.jpg'
img = np.array(Image.open(img))
print('image shape:', img.shape)  # (32, 32)
# Image.fromarray(img).show()

x = img.copy()
x1 = np.array(x).reshape(-1, 1)
print('x1 shape:', x1.shape)  # (1024, 1)
x1 = x1.reshape(-1)  # (1024,) 清晰图像
y1 = x1.copy()  # (1024,) 模糊图像

X = np.diag(np.abs(x1))
print('X shape:', X.shape)  # (1024, 1024)

X_inv = np.linalg.inv(X)
print(X_inv)

k = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0.0000, 0.0002, 0.0001, 0.0000, 0.0000, 0, 0, 0, 0,
              0, 0, 0, 0, 0.0002, 0.0157, 0.0057, 0.0001, 0.0000, 0.0000, 0.0000, 0, 0,
              0, 0, 0, 0, 0.0000, 0.0073, 0.0245, 0.0094, 0.0051, 0.0001, 0.0000, 0, 0,
              0, 0, 0.0000, 0.0000, 0.0000, 0.0000, 0.0178, 0.0927, 0.0935, 0.0012, 0.0001, 0, 0,
              0, 0, 0.0000, 0.0019, 0.0054, 0.0150, 0.0347, 0.1063, 0.0695, 0.0073, 0.0001, 0.0000, 0,
              0, 0.0000, 0.0056, 0.0296, 0.0311, 0.0242, 0.0230, 0.0293, 0.0134, 0.0047, 0.0002, 0.0002, 0,
              0, 0.0002, 0.0244, 0.0294, 0.0038, 0.0001, 0.0038, 0.0268, 0.0451, 0.0011, 0.0002, 0.0001, 0,
              0, 0.0005, 0.0263, 0.0158, 0.0001,  0.0001, 0.0004, 0.0193, 0.0599, 0.0294, 0.0002, 0.0000, 0,
              0, 0.0002, 0.0190, 0.0078, 0.0001, 0.0001, 0.0001, 0.0008, 0.0029, 0.0000, 0.0001, 0.0001, 0,
              0, 0.0000, 0.0020, 0.0043, 0.0000, 0.0001, 0, 0.0000, 0.0000, 0.0000, 0.0000, 0, 0,
              0, 0, 0.0000, 0.0000, 0.0000, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(13, 13)

# print('conv kernel:', k)
# print('original image:', x)
# print('image:', x1)

t = []

for i in range(k.shape[0]):
    t.append(np.append(k[i], np.zeros(19)))

z = []
for _ in range(img.shape[1] * (img.shape[0] - k.shape[0])):
    z.append(0)

z = np.array(z)
print(z.shape)

t = np.array(t).reshape(-1)
t = np.append(t, z)

np.set_printoptions(threshold=np.inf)
print(t)
K = []


def func(t1, n=1):
    res = []
    for _ in range(n):
        res.append(0)
    ll = 1024 - n
    res = np.array(res)
    res = np.append(res, t1[:ll])
    return np.array(res)


flag = 0
for i in range(img.shape[1] - k.shape[1] + 1):  # 20
    for j in range(img.shape[1] - k.shape[1] + 1):  # 20
        if flag == 0:
            K.append(func(t, 0))
            flag = 1
        else:
            K.append(func(t, 1))
    t = func(t, 32)
    flag = 0

print(len(K))  # (400, 1024)

y1 = y1.reshape(32, 32)
y1 = signal.convolve2d(y1, k, mode='valid')
y1 = y1.reshape(400, 1)

K = np.array(K)
A = X_inv + np.transpose(K).dot(K)
# print(X_inv)
x = x1.reshape(1024, 1)
# print(K[-1])
print(K.dot(x))
# x = np.ones((1024, 1))
b = np.transpose(K).dot(y1)


def test(A, x, b):
    r = b - np.dot(A, x)
    p = r
    for i in range(100):
        r1 = r
        a = np.dot(r.T, r) / np.dot(p.T, np.dot(A, p))
        x = x + a * p  # x(k+1)=x(k)+a(k)*p(k)
        r = b - np.dot(A, x)  # r(k+1)=b-A*x(k+1)
        q = np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b)
        if q < 10 ** -6:
            break
        else:
            beta = np.linalg.norm(r) ** 2 / np.linalg.norm(r1) ** 2
            p = r + beta * p  # p(k+1)=r(k+1)+beta(k)*p(k)
        return x


# x = test(A, x, b)

it = 10

for i in range(it):
    if np.linalg.det(A) == 0:
        x = np.linalg.pinv(A).dot(b)
    else:
        x = np.linalg.inv(A).dot(b)
    x1 = x.reshape(-1)
    X = np.diag(np.abs(x1))
    if np.linalg.det(X) == 0:
        X_inv = np.linalg.pinv(X)
    else:
        X_inv = np.linalg.inv(X)
    A = X_inv + np.transpose(K).dot(K)

# print(x)
print(x.shape)
print(np.min(x), np.max(x))
x = x.reshape(32, 32)
# x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
# print(x)
Image.fromarray(x).show()

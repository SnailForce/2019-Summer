import numpy as np
from PIL import Image
from scipy import signal
import scipy.misc

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
              0, 0.0005, 0.0263, 0.0158, 0.0001, 0.0001, 0.0004, 0.0193, 0.0599, 0.0294, 0.0002, 0.0000, 0,
              0, 0.0002, 0.0190, 0.0078, 0.0001, 0.0001, 0.0001, 0.0008, 0.0029, 0.0000, 0.0001, 0.0001, 0,
              0, 0.0000, 0.0020, 0.0043, 0.0000, 0.0001, 0, 0.0000, 0.0000, 0.0000, 0.0000, 0, 0,
              0, 0, 0.0000, 0.0000, 0.0000, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(13, 13)

# print('conv kernel:', k)
# print('original image:', x)
# print('image:', x1)
np.set_printoptions(threshold=np.inf)
K = []
r = np.zeros((32, 32))
for i in range(0, 20):
    for j in range(0, 20):
        r[i:i + 13, j:j + 13] = k
        r = r.reshape(-1)
        K.append(r)
        r = np.zeros((32, 32))

K = np.array(K)

t1 = np.transpose(K).dot(K)
print(t1.shape)
# print(t1)
print(K.shape)
y1 = y1.reshape(32, 32)
print(y1.shape, k.shape)
y1 = signal.convolve2d(y1, k, mode='valid')
h = Image.fromarray(y1)
scipy.misc.imsave('mohu.eps', y1)


y1 = y1.reshape(-1, 1)
t2 = np.transpose(K).dot(y1)
print(t2.shape)
A = X_inv + t1
# res = np.linalg.inv(A).dot(t2)
# print(res)


it = 100
b = t2
for i in range(it):
    if np.linalg.det(A) == 0:
        res = np.linalg.pinv(A).dot(b)
    else:
        res = np.linalg.inv(A).dot(b)
    x1 = res.reshape(-1)
    X = np.diag(np.abs(x1))
    if np.linalg.det(X) == 0:
        X_inv = np.linalg.pinv(X)
    else:
        X_inv = np.linalg.inv(X)
    A = X_inv + t1

res = res.reshape(32, 32)
Image.fromarray(res[6:26, 6:26]).show()
scipy.misc.imsave('qingxi3.eps', res[6:26, 6:26])
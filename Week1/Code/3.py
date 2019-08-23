import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import datasets
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D

data = datasets.load_digits()
x = data.data  # (1797, 64)
y = data.target  # (1797, )

# pca = decomposition.PCA(n_components=3)
# new_x = pca.fit_transform(x)  # (1797, 3)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)
#
# f = plt.figure()
# ax = f.add_subplot(111, projection='3d')
# ax.scatter(new_x[:, 0], new_x[:, 1], new_x[:, 2], c=y, cmap=cm.coolwarm)
# t = plt.gcf()
# t.savefig('2.eps', format='eps', dpi=1000)
# plt.show()

pca = decomposition.PCA(n_components=2)
new_x = pca.fit_transform(x)  # (1797, 3)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

f = plt.figure()
ax = f.add_subplot(111)
ax.scatter(new_x[:, 0], new_x[:, 1], c=y, cmap=cm.coolwarm)
t = plt.gcf()
t.savefig('3.eps', format='eps', dpi=1000)
plt.show()

# pca = decomposition.PCA(n_components=0.99)
# new_x = pca.fit_transform(x)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)
# print(pca.n_components_)

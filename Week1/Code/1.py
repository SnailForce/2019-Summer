import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([19, 45, 35, 31, 25, 32, 21, 26, 24, 27, 9, 23, 33, 29])
y = np.array([60, 113, 94, 90, 60, 88, 59, 61, 57, 78, 27, 72, 85, 63])
plt.scatter(x, y)
# plt.show()
print(np.corrcoef(x, y))  # 0.94

x, y = x.reshape(-1, 1), y.reshape(-1, 1)

lr_model = LinearRegression()
lr_model.fit(x, y)
print(lr_model.score(x, y))  # 0.887

alpha = lr_model.intercept_[0]
beta = lr_model.coef_[0][0]
print('alpha: ', alpha, '\nbeta: ', beta)  # 7.68 2.37
xx = np.arange(0, 70)
yy = beta * xx + alpha
plt.plot(xx, yy)

x_pre = np.array([15, 60]).reshape(-1, 1)
y_pre = lr_model.predict(x_pre)
plt.scatter(x_pre, y_pre)
plt.show()


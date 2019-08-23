import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
x = np.arange(1, 10, 0.01).reshape(-1, 1)
y = -0.51 * (x ** 5) + 15.04 * (x ** 4) - 176.38 * (x ** 3) + 1019.57 * (x ** 2) - 2907.62 * x + 3285.96
pol = PolynomialFeatures(degree=5)
lr = LinearRegression()
lr.fit(pol.fit_transform(x), y)
print(lr.coef_)
print(lr.intercept_)

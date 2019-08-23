from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

data = datasets.load_boston()
x = data.data[:, 5].reshape(-1, 1)
y = data.target.reshape(-1, 1)

x_train, x_train, y_train, y_train = train_test_split(x, y, train_size=0.25, random_state=0)

# lr_model = LinearRegression()
# lr_model.fit(x_train, y_train)
# y_pred = lr_model.predict(x_train)
#
# mse = mean_squared_error(y_train, y_pred)
# r = lr_model.score(x_train, y_train)
# print('LinearRegression: ')
# print('mse: ', mse, '\nscore: ', r)
#
# alpha = lr_model.intercept_[0]
# beta = lr_model.coef_[0]
# print('alpha: ', alpha, '\nbeta: ', beta)

xx = np.arange(min(x_train), max(x_train), 0.001)
# yy = beta * xx + alpha

# plt.scatter(x_train, y_train)
plt.scatter(x_train, y_train)
# plt.plot(xx, yy, label='1')
# plt.show()

xx = xx.reshape(-1, 1)

pol = PolynomialFeatures(degree=1)
x_train_pol = pol.fit_transform(x_train)
lr_model1 = LinearRegression()
lr_model1.fit(x_train_pol, y_train)
plt.plot(xx, lr_model1.predict(pol.transform(xx)), label='1', c='red')
# plt.show()
mse = mean_squared_error(y_train, lr_model1.predict(pol.transform(x_train)))
r = lr_model1.score(pol.transform(x_train), y_train)
print('LinearRegression1: ')
print('mse1: ', mse, '\nscore1: ', r)

alpha = lr_model1.intercept_[0]
beta = lr_model1.coef_[0]
print('alpha: ', alpha, '\nbeta: ', beta)


pol = PolynomialFeatures(degree=2)
x_train_pol = pol.fit_transform(x_train)
lr_model2 = LinearRegression()
lr_model2.fit(x_train_pol, y_train)
plt.plot(xx, lr_model2.predict(pol.transform(xx)), label='2', c='blue')
# plt.show()
mse = mean_squared_error(y_train, lr_model2.predict(pol.transform(x_train)))
r = lr_model2.score(pol.transform(x_train), y_train)
print('LinearRegression2: ')
print('mse2: ', mse, '\nscore2: ', r)

alpha = lr_model2.intercept_[0]
beta = lr_model2.coef_[0]
print('alpha: ', alpha, '\nbeta: ', beta)


pol = PolynomialFeatures(degree=3)
x_train_pol = pol.fit_transform(x_train)
lr_model3 = LinearRegression()
lr_model3.fit(x_train_pol, y_train)
mse = mean_squared_error(y_train, lr_model3.predict(pol.transform(x_train)))
r = lr_model3.score(pol.transform(x_train), y_train)
print('LinearRegression3: ')
print('mse3: ', mse, '\nscore3: ', r)
plt.plot(xx, lr_model3.predict(pol.transform(xx)), label='3', c='purple')
# plt.show()

alpha = lr_model3.intercept_[0]
beta = lr_model3.coef_[0]
print('alpha: ', alpha, '\nbeta: ', beta)


pol = PolynomialFeatures(degree=5)
x_train_pol = pol.fit_transform(x_train)
lr_model5 = LinearRegression()
lr_model5.fit(x_train_pol, y_train)
mse = mean_squared_error(y_train, lr_model5.predict(pol.transform(x_train)))
r = lr_model5.score(pol.transform(x_train), y_train)
print('LinearRegression5: ')
print('mse5: ', mse, '\nscore5: ', r)
plt.plot(xx, lr_model5.predict(pol.transform(xx)), label='5', c='black')

alpha = lr_model5.intercept_[0]
beta = lr_model5.coef_
print('alpha: ', alpha, '\nbeta: ', beta)


pol = PolynomialFeatures(degree=10)
x_train_pol = pol.fit_transform(x_train)
lr_model10 = LinearRegression()
lr_model10.fit(x_train_pol, y_train)
mse = mean_squared_error(y_train, lr_model10.predict(pol.transform(x_train)))
r = lr_model10.score(pol.transform(x_train), y_train)
print('LinearRegression10: ')
print('mse10: ', mse, '\nscore10: ', r)
plt.plot(xx, lr_model10.predict(pol.transform(xx)), label='10', c='green')

alpha = lr_model10.intercept_[0]
beta = lr_model10.coef_[0]
print('alpha: ', alpha, '\nbeta: ', beta)

plt.legend()
plt.xlabel('rooms')
plt.ylabel('price')
t = plt.gcf()
t.savefig('1.eps', format='eps', dpi=1000)
plt.show()

'''
LinearRegression: 
mse:  43.472041677202206 
score:  0.46790005431367815
alpha:  -36.180992646339206 
beta:  9.312949225629252
LinearRegression2: 
mse2:  37.03687297431593 
score2:  0.5466668383242808
LinearRegression3: 
mse3:  37.100463580083655 
score3:  0.5458884861565492
LinearRegression5: 
mse5:  38.24695342979688 
score5:  0.5318553935474608
LinearRegression10: 
mse10:  41.278725136516414 
score10:  0.49474635752708007
'''

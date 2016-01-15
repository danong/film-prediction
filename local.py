import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# columns: year, length, budget, rating, action, animation, comedy, drama, documentary, romance, short
data = np.genfromtxt('movies.tab', comments = '\\', delimiter = '\t', skip_header = 1, usecols = (0,1,2,3,4,17,18,19,20,21,22,23) )

# remove rows with missing data. how does this work?!
data = data[~np.isnan(data).any(axis=1)]


# create X and Y
data_X = np.delete(data, 3, 1)
data_y = data[:, 3]

poly = PolynomialFeatures(2)
data_X = poly.fit_transform(data_X)

# Split into training/testing
data_X_train = data_X[:-800]
data_X_test = data_X[-800:]
data_y_train = data_y[:-800]
data_y_test = data_y[-800:]

print("data_X_train: [", data_X_train.shape, "]")
print("data_X_test: [", data_X_test.shape, "]")
print("data_y_train: [", data_y_train.shape, "]")
print("data_y_test: [", data_y_test.shape, "]")

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data_X_train, data_y_train)


print(regr.predict(data_X_test))
print(data_y_test)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f" % np.mean((regr.predict(data_X_test) - data_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(data_X_test, data_y_test))

# plt.scatter(data_X_test, data_y_test,  color='black')
# plt.plot(data_X_test, regr.predict(data_X_test), color='blue',
         # linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
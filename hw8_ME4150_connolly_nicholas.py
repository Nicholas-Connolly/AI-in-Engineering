# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:30:19 2019

@author: conno
"""

# ME:4150 Artificial Intelligence in Engineering
# Concrete analysis homework
# Orignated by Prof. Shaoping Xiao

# Import basic libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso



print('Part 1: Defining a regression problem.')

# Import concrete data set.
print('\nRead in data set from excel file')
#read from excel file
concrete = pd.read_excel("concrete.xls")

print('\nConcrete data set, first five rows:\n')
print(concrete.head())


print('\n\nDefining regression problem:')
print('\nUsing X = age and y = Cstr')
X = concrete['age'].values
y = concrete['Cstr'].values

print('\nPlotting raw data\n')
plt.scatter(X, y, marker = '+', c='k')
plt.xlabel('Concrete Age (X) days', size=15)
plt.ylabel('Compressive Strength (y) MPa', size=15)
plt.title('Concrete Age vs. Strength', size= 15)
plt.show()



print('\n\nPart 2: Using a polynomial regression model (and comparing with linear model)')

# training/validation/test sets splitting
print('\nSplitting data into training/validation/test sets.')
X=X.reshape(-1,1)   # Samples are in column vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 0)

print('\n')
print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the validation set.'.format(len(y_valid)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))
print('\n')

# linear regression
print('Creating a linear regression model')
linreg = LinearRegression()     # initialization and configuration
linreg.fit(X_train, y_train)    # training
print('The regression function is y=({:.3f})X + ({:.3f})'
     .format(linreg.coef_[0], linreg.intercept_))
print('R-squared score: {:.3f}(training) and {:.3f}(validation) '
     .format(linreg.score(X_train, y_train), linreg.score(X_valid, y_valid)))


print('\n\nCreating a polynomial regression model')

# adding polynomial features 
print('\nAdding polynomial features, choosing degree 12 polynomial')
ndegree = 12
poly = PolynomialFeatures(degree = ndegree)

X_train_poly = poly.fit_transform(X_train)
X_valid_poly = poly.fit_transform(X_valid)
X_test_poly = poly.fit_transform(X_test)

# Feature scaling 
print('\nScaling features')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_valid_scaled = scaler.transform(X_valid_poly)
X_test_scaled = scaler.transform(X_test_poly)


# linear regression for polynomial
linreg_poly = LinearRegression().fit(X_train_scaled, y_train)
np.set_printoptions(precision=2)    # keep 2 digits when printing a float array
print('The coefficients of this polynomial are b=({:.2f}), \n w=({})'
     .format(linreg_poly.intercept_, linreg_poly.coef_))
print('\n')
print('R-squared score: {:.3f}(training) and {:.3f}(validation) '
     .format(linreg_poly.score(X_train_scaled, y_train), linreg_poly.score(X_valid_scaled, y_valid)))


# plot both the linear function and the polynomial  
print('\n\nPlotting both linear and polynomial regression models')
X_plot=np.arange(X_train.min(), X_train.max(), 0.1)
X_plot.shape=(X_plot.size,1)
X_plot_poly = poly.fit_transform(X_plot)
X_plot_scaled = scaler.transform(X_plot_poly)
y_plot=linreg_poly.predict(X_plot_scaled)

plt.figure()
plt.scatter(X_valid, y_valid, marker= 'o', c='red', s=50, label='test set')
plt.plot(X_plot, linreg.predict(X_plot), 'b-', label='Linear regression')
plt.plot(X_plot, y_plot, 'g-', label='poly degree {}'.format(ndegree), linewidth =3)
plt.legend(loc='best', scatterpoints=1,  fontsize=15, frameon=False, labelspacing=0.5)
plt.title('Poly regression vs. linear regresion', size=15)
plt.xlabel('Concrete Age (X) days', size=15)
plt.ylabel('Compressive Strength (y) MPa', size=15)
plt.show()

print('\nRescaling plot to view the absurd overfitting of the default polynomial model\n')
plt.figure()
plt.scatter(X_valid, y_valid, marker= 'o', c='red', s=50, label='test set')
plt.plot(X_plot, linreg.predict(X_plot), 'b-', label='Linear regression')
plt.plot(X_plot, y_plot, 'g-', label='poly degree {}'.format(ndegree), linewidth =3)
plt.legend(loc='best', scatterpoints=1,  fontsize=10, frameon=False, labelspacing=0.5)
plt.title('Poly regression vs. linear regresion\nRESCALED', size=15)
plt.xlabel('Concrete Age (X) days', size=15)
plt.ylabel('Compressive Strength (y) MPa', size=15)
plt.ylim(0,100)
plt.show()

print('\nNOTE: I will avoid including the non-regularized polynomial plot in future graphs')




print('\n\nPart 3: Using a polynomial regression model with regularization')
print('(I will use both ridge and lasso)')

# Ridge regularization
print('\nUsing ridge regularization:\n')
alpha_ = 0.1
linridge = Ridge(alpha=alpha_).fit(X_train_scaled, y_train)


print('Without regularization, the coefficients of this polynomial are \n b=({:.2f}), \n w=({})'
     .format(linreg_poly.intercept_, linreg_poly.coef_))
print('R-squared score: {:.3f}(training) and {:.3f}(validation) '
     .format(linreg_poly.score(X_train_scaled, y_train), linreg_poly.score(X_valid_scaled, y_valid)))

print('With regularization, the coefficients of this polynomial are \n b=({:.2f}), \n w=({})'
     .format(linridge.intercept_, linridge.coef_))
print('R-squared score: {:.3f}(training) and {:.3f}(validation) '
     .format(linridge.score(X_train_scaled, y_train), linridge.score(X_valid_scaled, y_valid)))

# plot ridge regularization model 
print('\nPlotting polynomial regression model with ridge regularization')
y_plot_ridge=linridge.predict(X_plot_scaled)

plt.figure()
plt.scatter(X_valid, y_valid, marker= 'o', c='red', s=50, label='validation set')
#plt.plot(X_plot, y_plot, 'g-', label='w/o regularization (n={})'.format(ndegree), linewidth =3)
plt.plot(X_plot, y_plot_ridge, 'b-', label='Ridge regularization', linewidth =3)
plt.legend(loc='best', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.title('Ridge Regularization ($\\alpha$ = {}) \n training score ({:.3f}) and validation score ({:.3f})'
          .format(alpha_, linridge.score(X_train_scaled, y_train), 
                  linridge.score(X_valid_scaled, y_valid)), size=15)
plt.xlabel('Concrete Age (X) days', size=15)
plt.ylabel('Compressive Strength (y) MPa', size=15)
plt.show()

# Lasso regularization
print('\nUsing lasso regularization:\n')
alpha_la = 0.01
nmax = 1000000
linlasso = Lasso(alpha=alpha_la, max_iter = nmax).fit(X_train_scaled, y_train)


print('Without regularization, the coefficients of this polynomial are \n b=({:.2f}), \n w=({})'
     .format(linreg_poly.intercept_, linreg_poly.coef_))
print('R-squared score: {:.3f}(training) and {:.3f}(validation) '
     .format(linreg_poly.score(X_train_scaled, y_train), linreg_poly.score(X_valid_scaled, y_valid)))

print('With lasso regularization, the coefficients of this polynomial are \n b=({:.2f}), \n w=({})'
     .format(linlasso.intercept_, linlasso.coef_))
print('R-squared score: {:.3f}(training) and {:.3f}(validation) '
     .format(linlasso.score(X_train_scaled, y_train), linlasso.score(X_valid_scaled, y_valid)))

# plot ridge regularization model
print('\nPlotting polynomial regression model with lasso regularization')
y_plot_lasso=linlasso.predict(X_plot_scaled)

plt.figure()
plt.scatter(X_valid, y_valid, marker= 'o', c='red', s=50, label='validation set')
#plt.plot(X_plot, y_plot, 'g-', label='w/o regularization (n={})'.format(ndegree), linewidth =3)
plt.plot(X_plot, y_plot_lasso, 'b-', label='Lasso regularization', linewidth =3)
plt.legend(loc='best', scatterpoints=1,  fontsize=12, frameon=False, labelspacing=0.5)
plt.title('Lasso Regularization ($\\alpha$ = {}) \n training score ({:.3f}) and validation score ({:.3f})'
          .format(alpha_la, linlasso.score(X_train_scaled, y_train), 
                  linlasso.score(X_valid_scaled, y_valid)), size=15)
plt.xlabel('Concrete Age (X) days', size=15)
plt.ylabel('Compressive Strength (y) MPa', size=15)
plt.show()





# Discussion of Results
print('\n\nPart 4: Discussion and model comparison (in Python code comments)')

"""
Even though the age of concrete in the data ranges from 0 to 365 days,
it is clear from the first plot that age takes on only about 10 different values.
Even so, there does appear to be a polynomial trend in how the data is clustered.

The default polynomial regression model without regularization has extreme overfitting.
This appears to be the result of the large gaps in the available ages of concrete.
By using absurdly large coefficients, it fits a polynomial through the vertical clusters of data points.
Adding regularization to the model eliminates this without a significant loss in the training/valdiation score.

Using ridge or lasso regularization yield approximately the same training/validation score of about one third.
This score seems especially low when compared with the scores of previous binary classification models,
but the polynomial regression plots with regularization do appear to capture the basic trend of the data.

It is worth noting that the first two coefficients in the weight vector are approximately equal in both models.
The higher order terms differ, but are relatively small by comparison.
This explains the similarity in the plots of each regularization model.

"""






















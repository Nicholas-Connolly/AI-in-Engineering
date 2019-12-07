# ME:4150 Artificial Intelligence in Engineering
# Concrete analysis homework
# Orignated by Prof. Shaoping Xiao

# PART 0: import basic libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC



# PART 1: read in data set, summarize first few rows
print('\nPart 1: read in data set from excel file')
#read from excel file
concrete = pd.read_excel("concrete.xls")

# here is the datastructure of glass
print('\nConcrete data set, first five rows:\n')
print(concrete.head())



# PART 2: convert to a binary classification problem
print('\n\nPart 2: convert compressive strength to a binary classification problem')
print('Using compressive strength > 35MPa as true class and < 35MPa as false')
y_ = concrete['Cstr']
y = y_ > 35  # compressive strength larger than 35MPa is the true class



# PART 3: data pre-processing
print('\n\nPart 3: data pre-processing')
# Sub-Part a: summarize statistical properties of each feature
print('\nPart 3a: Summarizing statistical properties of features in data set')
print('\nConcrete data set, statistics summary:\n')
for x in concrete: 
    c=concrete[x]
    print(x, ' min:{:.2f}  max:{:.2f}  mean:{:.2f}  std:{:.2f}'.format(c.min(),c.max(),c.mean(),c.std()))

    
# Sub-Part b: output the class size for each of the two binary classes.
print('\nPart 3b: Counting the number of data samples in each class\n')
print('The class of compressive strength larger than 35MPa has', sum(e==1 for e in y), 'data samples')
print('The class of compressive strength smaller than 35MPa has', sum(e==0 for e in y), 'data samples')


# Sub-Part c: Choose four features to visualize their distributions and correlations
print('\nPart 3c: Visualing feature distributions and correlations')
X = concrete[['cement', 'Water', 'SP', 'age']]
print('\nVisual summary of concrete features: cement, water, superplasticizer, and age')
Axes=pd.plotting.scatter_matrix(X, c= y, marker = 'o', s=40, figsize=(12,12))
[plt.setp(item.yaxis.get_label(), 'size', 20) for item in Axes.ravel()]
[plt.setp(item.xaxis.get_label(), 'size', 20) for item in Axes.ravel()]
plt.show()


# Sub-Part d: Select two primary features to conduct classification
print('\nPart 3d: Selecting primary features for classification')
print('\nVisualizing entire concrete data set using only cement and water')
X = concrete[['cement', 'Water']].values

# visualize the whole dataset
x1_min, x1_max = X[:, 0].min() - 50, X[:, 0].max() + 50
x2_min, x2_max = X[:, 1].min() - 25, X[:, 1].max() + 75

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X[y==True,0],X[y==1,1], marker = 'o', c='r', label='compressive strength > 35MPa')
plt.scatter(X[y==False,0],X[y==0,1], marker = 'o', c='b', label='compressive strength < 35MPa')
plt.xlabel('concrete feature cement (x1)', size=15)
plt.ylabel('concrete feature water (x2)', size=15)
plt.title('Compressive Strength: >35MPA vs <35MPA', size= 15)
plt.legend(loc='best', fontsize='large')
plt.show()



# PART 4: model training
print('\n\nPart 4: model training')

#Split the data set into the training and test sets
print('\nSplitting concrete data set into training and test sets\n')
X_train_, X_test_, y_train, y_test = train_test_split(X,y, random_state = 4)

print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))

print('\nStandardizing data for convenience')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_)
X_test = scaler.transform(X_test_)

print('\nVisualizing standardized training data')
x1_min, x1_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
x2_min, x2_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X_train[y_train==True,0],X_train[y_train==1,1], marker = 'o', c='r', label='> 35MPa')
plt.scatter(X_train[y_train==False,0],X_train[y_train==0,1], marker = 'o', c='b', label='< 35MPa')
plt.xlabel('concrete feature cement (x1)', size=15)
plt.ylabel('concrete feature water (x2)', size=15)
plt.title('Concrete compressive strength training data (standardized)', size= 15)
plt.legend(loc='best', fontsize='large')
plt.show()


# List models
print('\n\nTraining classification models: logistic regression, LSVC, and SVC with rbf kernel')

# Logisitic model from preceding homework
print('\n\nMODEL 1: Logistic Regression')
print('Creating logisitc regression model (same as from homework 6)')
print('\nUsing 5-fold cross validation to select the best hyper parameters for logistic model\n')

C_array = np.arange(0.01, 5.01, 0.01)
best_score = 0.0

for C_logistic in C_array:
    clf = LogisticRegression(C=C_logistic, solver='lbfgs')
    valid_score = cross_val_score(clf, X_train, y_train, cv=5)
    if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_C = C_logistic

print('The best validation score: {:.2f}%'.format(best_score*100))
print('The best parameter C: {}'.format(best_C))


# Train the best logisitc model with the training set
print('\nTraining best logisitic model, visualizing regression\n')
logclf = LogisticRegression(C=best_C, solver='lbfgs')   # initialization and configuration
logclf.fit(X_train, y_train)    # training
print('The training score: {:.2f}%'
     .format(logclf.score(X_train, y_train)*100))
print('The test score: {:.2f}%'
     .format(logclf.score(X_test, y_test)*100))
TestScoreLogistic = logclf.score(X_test, y_test)*100


print('\nConcrete Compressive Strength Logistic decision boundary \n ({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0'
          .format(logclf.intercept_[0], logclf.coef_[0,0], logclf.coef_[0,1]))

# Model evaluation via the test set
x1_plot = np.arange(x1_min, x1_max, (x1_max-x1_min)/100.0)
x1_plot = x1_plot.reshape(-1, 1)
x2_plot = -(logclf.coef_[0,0]*x1_plot + logclf.intercept_[0])/logclf.coef_[0,1]
plt.plot(x1_plot, x2_plot, '-', c='black', label='Logistic decision boundary')

y_predict = logclf.predict(X_test)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X_test[y_predict==1,0],X_test[y_predict==1,1], marker = '+', c='r', s=200, label='> 35MPa prediction')
plt.scatter(X_test[y_predict==0,0],X_test[y_predict==0,1], marker = '+', c='b', s=200, label='< 35MPa prediction')
plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1], marker = 'o', c='r', s=30, label='> 35MPa test')
plt.scatter(X_test[y_test==0,0],X_test[y_test==0,1], marker = 'o', c='b', s=30, label='< 35MPa test')
plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5)
plt.title('Concrete Compressive Strength vs Logistic prediction \n decision boundary ({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0'
          .format(logclf.intercept_[0], logclf.coef_[0,0], logclf.coef_[0,1]), size=15)
plt.xlabel('concrete feature cement (x1)', size=15)
plt.ylabel('concrete feature water (x2)', size=15)
plt.legend(loc='best', fontsize='medium')
plt.show()


print('\nRecording confusion matrix and performance metrics for logistic model')
ConfmatLogistic = confusion_matrix(y_true=y_test, y_pred=y_predict)
PrecisionLogistic = precision_score(y_true=y_test, y_pred=y_predict)
RecallLogistic = recall_score(y_true=y_test, y_pred=y_predict)
F1Logistic = f1_score(y_true=y_test, y_pred=y_predict)





# LSVC Model
print('\n\nMODEL 2: Linear Support Vector Classifier')
print('Creating LSVC model')
print('\nUsing 5-fold cross validation to select the best hyper parameters for LCSV model\n')

C_array = np.arange(0.01, 5.01, 0.01)
best_score = 0.0

for C_lsvc in C_array:
    clf = LinearSVC(C=C_lsvc, max_iter=1000000) 
    clf.fit(X_train, y_train)
    valid_score = cross_val_score(clf, X_train, y_train, cv=5)
    if valid_score.mean() > best_score:
        best_score = valid_score.mean()
        best_C = C_lsvc

print('The best validation score: {:.2f}%'.format(best_score*100))
print('The best parameter C: {}'.format(best_C))


print('\nTraining LSVC model using the best parameter\n')
clf = LinearSVC(C=best_C, max_iter=1000000)   # initialization and configuration
clf.fit(X_train, y_train)    # training
print('The training score: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('The test score: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))
TestScoreLSVC = clf.score(X_test, y_test)*100

# Decision boundary
print('\nConcrete Compressive Strength LSVC decision boundary \n ({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0'
          .format(clf.intercept_[0], clf.coef_[0,0], clf.coef_[0,1]))

# Model evaluation via the test set
x1_plot = np.arange(x1_min, x1_max, (x1_max-x1_min)/100.0)
x1_plot = x1_plot.reshape(-1, 1)
x2_plot = -(clf.coef_[0,0]*x1_plot + clf.intercept_[0])/clf.coef_[0,1]
plt.plot(x1_plot, x2_plot, '-', c='black', label='LSVC decision boundary')

y_predict = clf.predict(X_test)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X_test[y_predict==1,0],X_test[y_predict==1,1], marker = '+', c='r', s=200, label='> 35MPa prediction')
plt.scatter(X_test[y_predict==0,0],X_test[y_predict==0,1], marker = '+', c='b', s=200, label='< 35MPa prediction')
plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1], marker = 'o', c='r', s=30, label='> 35MPa test')
plt.scatter(X_test[y_test==0,0],X_test[y_test==0,1], marker = 'o', c='b', s=30, label='< 35MPa test')
plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5)
plt.title('Concrete Compressive Strength vs LSVC prediction \n decision boundary ({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0'
          .format(clf.intercept_[0], clf.coef_[0,0], clf.coef_[0,1]), size=15)
plt.xlabel('concrete feature cement (x1)', size=15)
plt.ylabel('concrete feature water (x2)', size=15)
plt.legend(loc='best', fontsize='medium')
plt.show()


print('\nRecording confusion matrix and performance metrics for LSVC model')
ConfmatLSVC = confusion_matrix(y_true=y_test, y_pred=y_predict)
PrecisionLSVC = precision_score(y_true=y_test, y_pred=y_predict)
RecallLSVC = recall_score(y_true=y_test, y_pred=y_predict)
F1LSVC = f1_score(y_true=y_test, y_pred=y_predict)





# SVC Model (with rbf kernel)
print('\n\nMODEL 3: Support Vector Classifier with Radial Basis Function kernel')
print('Creating SVC model with rbf kernel')
print('\nUsing 5-fold cross validation to select the best hyper parameters for SVC model\n')


# SVC with rbf kerenel classifier training and cross-validation
best_score = 0.0
gamma_array = np.arange(0.1, 5.1, 0.1)
C_array = np.arange(0.1, 5.1, 0.1)

for gamma_svc in gamma_array:
    for C_svc in C_array:
        clf = SVC(kernel='rbf', random_state=0, gamma=gamma_svc, C=C_svc)
        valid_score = cross_val_score(clf, X_train, y_train, cv=5)
        if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_parameters = {'C':C_svc, 'gamma':gamma_svc}

print('The best validation score: {:.3f}'.format(best_score))
print('The best parameters: {}'.format(best_parameters))

# retrain the best model
print('\nRetraining SVC (with rbf kernel) model using best parameters\n')
clf = SVC(kernel='rbf', random_state=0, **best_parameters)
clf.fit(X_train, y_train)

print('Accuracy of rbf SVC classifier on the training set: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('Accuracy of rbf SVC classifier on the test set: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))
TestScoreSVC = clf.score(X_test, y_test)*100

# Plot the decision region vs test data
plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

h = .02  # spacing between grid points
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# plot the results in a contour plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(5, 4))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# plot validation set vs its prediction
y_predict = clf.predict(X_test)

plt.scatter(X_test[y_predict==1,0],X_test[y_predict==1,1], marker = '+', c='r', s=200, label='> 35MPa prediction')
plt.scatter(X_test[y_predict==0,0],X_test[y_predict==0,1], marker = '+', c='b', s=200, label='< 35MPa prediction')
plt.scatter(X_test[y_test==1,0],X_test[y_test==1,1], marker = 'o', c='r', s=30, label='> 35MPa test')
plt.scatter(X_test[y_test==0,0],X_test[y_test==0,1], marker = 'o', c='b', s=30, label='< 35MPa test')
plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5)
plt.title('Concrete Compressive Strength vs rbf SVC \n test score = {:.2f}% (gamma={:.2f} and C={:.2f}) '
          .format((clf.score(X_test, y_test)*100), best_parameters['gamma'], best_parameters['C']), size=15)
plt.xlabel('concrete feature cement (x1)', size=15)
plt.ylabel('concrete feature water (x2)', size=15)
plt.legend(loc='best', scatterpoints=1, numpoints=1)
plt.show()

print('\nRecording confusion matrix and performance metrics for LSCV model')
ConfmatSVC = confusion_matrix(y_true=y_test, y_pred=y_predict)
PrecisionSVC = precision_score(y_true=y_test, y_pred=y_predict)
RecallSVC = recall_score(y_true=y_test, y_pred=y_predict)
F1SVC = f1_score(y_true=y_test, y_pred=y_predict)




# Part 5: Final Evaluation
print('\n\nPart 5: Comparing all three models')

# Logistic model
print('\n\nDisplaying performance details for logisitc model\n')
print('Test Score: %.3f' % TestScoreLogistic)
print('\nConfusion Matrix:')
print(ConfmatLogistic, '\n')
print('Precision: %.3f' % PrecisionLogistic)
print('Recall: %.3f' % RecallLogistic)
print('F1 score: %.3f' % F1Logistic)

# LSVC model
print('\n\nDisplaying performance details for LSVC model\n')
print('Test Score: %.3f' % TestScoreLSVC)
print('\nConfusion Matrix:')
print(ConfmatLSVC, '\n')
print('Precision: %.3f' % PrecisionLSVC)
print('Recall: %.3f' % RecallLSVC)
print('F1 score: %.3f' % F1LSVC)

# SVC with rbf kernel model
print('\n\nDisplaying performance details for SVC (with rbf kernel) model\n')
print('Test Score: %.3f' % TestScoreSVC)
print('\nConfusion Matrix:')
print(ConfmatSVC, '\n')
print('Precision: %.3f' % PrecisionSVC)
print('Recall: %.3f' % RecallSVC)
print('F1 score: %.3f' % F1SVC)


print('\n\nConclusion: based on test data performance, the rbf SVC model is the most effective classifier')


# Discussion of Results
print('\n\nDetailed discussion and model comparison (in Python code comments)')

"""
The performance of each of the three models on the test set is:
    65.9% for the Logistic model;
    67.1% for the LSVC model;
    73.3% for the SVC (with rbf  kernel) model.
Based on test data performance alone, the SVC model is the most acurate.
This is to be expected since it can capture some of the nonlinearity in the data.
The LSVC model performs slightly better than the Logistic model, but both models have similar decision boundaries.

In terms of the confusion matrix, recall that these matrices are organized as:
    [[ TN FP ]
     [ FN TP ]]
Hence, the main daigonal entries refer to correct predictions while the other diagonal is incorrect predictions.
Both the Logistic and LSVC models have very similar confusion matrices, with entries differing by no more than 4.
This also means that they have approximately equal Precision, Recall, and F1 score.
This is to be expected since both models are linear with an approximately equal decision boundary.

By contrast, the confusion matrix for the SVC model has overall more correct predictions,
but it is interesting to note that this model has fewer TPs and more FNs than the two linear models.
This model only sees real improvement in the number of TNs and FPs (the top row of the confusion matrix).
In other words, the SVC model is effective at correctly identifing the false class in this problem.

Interestingly, the SVC model is less effective at correctly identifying the true class than either linear model.
However, the difference is essentially negligible since it only falls short by a few predictions.

All three models have comparable Recall and F1 scores, but the SVC model has the highest precision score.
It exceeds the precision of the other two models by more than 10%.
Notice that the SVC model has many fewer FPs than the other two models, but a similar number of TPs.
Since precision is defined by TP/(TP+FP), this higher value for the SVC model is expected.

Ultimately, we may conclude that the rbf SVC model is the most effective of the three for classifying
the compressive strength of the concrete data set in terms of the two chosen features of cement and water.

"""









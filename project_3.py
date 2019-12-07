# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score



print('\nReading "seeds.txt" data set\n')

# Import Data Set
seeds = pd.read_excel('seeds_datasets.xlsx')
print(seeds.head())

# Divide data set into features and classes
X = seeds[['Area','Perimeter','Compactness','Length','Width','Asymmetry','Groove']]

#y_ = seeds['Class']
y = seeds['Class']

#y4 = np.zeros(len(seeds))
#for i in range (0, len(seeds), 1):
#    y4[i] = y4_[i]

# Declare arrays of feature names and class names
seeds_feature_names = ['Area','Perimeter','Compactness','Length','Width','Asymmetry','Groove']
seeds_target_names = ['Kama','Rosa','Canadian']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)



# Use a decision tree to identify the two most important features
# decision tree cross-validation
d_array = np.arange(1, 7, 1)  #max_depth
best_score = 0.0

for tdepth in d_array:
    clf = DecisionTreeClassifier(max_depth = tdepth) 
    valid_score = cross_val_score(clf, X_train, y_train, cv=5)
    if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_d = tdepth

print("\nThe best max-depth for decision trees is {:2d}".format(best_d))

# retrain the best model
clf = DecisionTreeClassifier(max_depth = best_d)
clf.fit(X_train, y_train)    # training
print('\n')
print('Accuracy of Decision tree classifier on the training set: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('Accuracy of Decision tree classifier on the test set: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))
y_pred = clf.predict(X_test)


# multi-class classification confusion matrix
confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('Decision Tree test score: {0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()


# multi-class classification report

print(classification_report(y_test, y_pred, target_names=seeds_target_names))
print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))


# Identify which features are the most important.
importances = clf.feature_importances_
fnames = np.array(seeds_feature_names)
indices = np.argsort(importances)

plt.figure()
plt.title('Feature importances (max_depth={}) \n {}'.format(best_d, importances), size=15)
plt.barh(range(X.shape[1]), importances[indices], color="b")
plt.yticks(range(X.shape[1]), fnames[indices])
plt.yticks(range(7))
plt.ylim([-1, X.shape[1]])
plt.xlabel('Importance', size=15)
plt.ylabel('Features', size=15)
plt.show()


# Select the two most important features, ordered by importance
index1 = indices[-1]
index2 = indices[-2]
feature1 = seeds_feature_names[index1]
feature2 = seeds_feature_names[index2]
print('Two most import features: '+feature1+' and '+feature2)

# Reducing extracting these two columns from the original data set.
X_reduced = seeds[[feature1,feature2]].values

# Visualing data
print('\nVisualizing seeds data set using two most important features')
colors = ['r', 'b', 'y']
plt.figure()
for yy in range(1, 4):
    plt.scatter(X_reduced[y==yy, 0], X_reduced[y==yy, 1], marker='o', c=colors[yy-1],
                label=seeds_target_names[yy-1])
plt.xlabel('Seeds Feature '+feature1+' (x1)', size=15)
plt.ylabel('Seeds Feature '+feature2+' (x2)', size=15)
plt.title('Seeds Data', size= 15)
plt.legend(loc='best',fontsize='large')
plt.show()




# Resplit the seeds data, now using the reduced data set with the two most important features.
X_train, X_test, y_train, y_test = train_test_split(X_reduced,y, test_size = 0.2, random_state = 4)

print('\nCreating a multi-classification model using logistic regression')
print('Using 5-fold cross validation to search for the best hyper-parameter')
C_array = np.arange(0.1, 5.1, 0.1)
best_score = 0.0

for C_logistic in C_array:
    #clf = LinearSVC(C=C_lsvc, max_iter=100000) # initialization and configuration
    clf = LogisticRegression(C=C_logistic, solver='lbfgs', multi_class='ovr') #NOTE: 'ovr' denotes One-vs.-Rest
    valid_score = cross_val_score(clf, X_train, y_train, cv=5)
    if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_C = C_logistic

print("\nThe best C is ", best_C)
print('Retraining logistic regression model using best hyper-parameter\n')

# retrain the best model
#clf = LinearSVC(C=best_C, max_iter=100000)
clf = LogisticRegression(C=best_C, solver='lbfgs', multi_class='ovr') #NOTE: 'ovr' denotes One-vs.-Rest
clf.fit(X_train, y_train)    # training
print('The training score: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('The test score: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))

print("\n")
print("Logistic regression decision boundaries are:")
for w, b in zip(clf.coef_, clf.intercept_):
    print( "({:.2f})+({:.2f})(x1)+({:.2f})(x2)=0".format(b, w[0], w[1]))
        
# plot decision boundaries, the testing set and its prediction
x1_min, x1_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
x2_min, x2_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5

x_plot = np.arange(x1_min, x1_max, (x1_max-x1_min)/100.0)
y_pred = clf.predict(X_test)

plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', 
                c=colors[yy-1], label=seeds_target_names[yy-1])

for yy in range(1, 4):
    plt.scatter(X_test[y_pred==yy, 0], X_test[y_pred==yy, 1], marker='+', s=200, 
                c=colors[yy-1], label=seeds_target_names[yy-1])

for w, b, color in zip(clf.coef_, clf.intercept_, colors):
    plt.plot(x_plot, -(x_plot * w[0] + b) / w[1], c=color, alpha=0.8, linewidth=3)
    
plt.title('Logistic Regression test data (.) vs prediction (+): \n test score = {:.2f}%'.
          format(clf.score(X_test, y_test)*100), size=15)
plt.xlabel('Seeds Feature '+feature1+' (x1)', size=15)
plt.ylabel('Seeds Feature '+feature2+' (x2)', size=15)
plt.legend(loc='best',fontsize='large')
plt.show()

# decision regions: a contour plot
plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

h = .02  
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', c=colors[yy-1], 
                label=seeds_target_names[yy-1])
plt.title('Logistic Regression decision regions and the test set', size=15)
plt.xlabel('Seeds Feature '+feature1+' (x1)', size=15)
plt.ylabel('Seeds Feature '+feature2+' (x2)', size=15)
plt.legend(loc='best',fontsize='large')
plt.show()
print("\n")

# multi-class classification confusion matrix
confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('Logistic Regression Accuracy: {0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()


# multi-class classification report
print(classification_report(y_test, y_pred, target_names=seeds_target_names))

print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))
print("\n")

###############################################################################

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

print("\n")
print('Best parameters: {}'.format(best_parameters))
print("\n")

# retrain the best model
clf = SVC(kernel='rbf', random_state=0, **best_parameters)
clf.fit(X_train, y_train)

print('Accuracy of SVC with RBF Kernel classifier on the training set: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('Accuracy of SVC with RBF Kernel classifier on the test set: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))
print('\n')

# multi-class classification confusion matrix
y_pred = clf.predict(X_test)
confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('SVC with RBF Kernel Accuracy:{0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()

plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

h = .02  
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', c=colors[yy-1], 
                label=seeds_target_names[yy-1])
plt.title('SVC with RBF Kernel decision regions and the test set', size=15)
plt.xlabel('Seeds Feature '+feature1+' (x1)', size=15)
plt.ylabel('Seeds Feature '+feature2+' (x2)', size=15)
plt.legend(loc='best',fontsize='large')
plt.show()


# multi-class classification report
print(classification_report(y_test, y_pred, target_names=seeds_target_names))
print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))

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
print('\n\nPart 4: training a logistic regression model')

# Sub-Part a: Split the data set into the training and test sets
print('\nPart 4a: Splitting concrete data set into training and test sets\n')
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


# Sub-Part b: Apply logistic regression
print('\nPart 4b: Applying logistic regression (using default hyper parameters)\n')

logclf = LogisticRegression(C=1.0, solver='lbfgs')   # initialization and configuration
logclf.fit(X_train, y_train)    # training
print('Accuracy of Logistic regression classifier on the training set: {:.2f}%'
     .format(logclf.score(X_train, y_train)*100))
print('Accuracy of Logistic regression classifier on the test set: {:.2f}%'
     .format(logclf.score(X_test, y_test)*100))


# Sub-Part c: Apply cross-validation to select the best model
print('\n\nPart 4c: Using 5-fold cross validation to select the best hyper parameters\n')

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


# Sub-Part d: Train the best model with the training set
print('\n\nPart 4d: Training best model, visualizing regression\n')
logclf = LogisticRegression(C=best_C, solver='lbfgs')   # initialization and configuration
logclf.fit(X_train, y_train)    # training
print('The training score: {:.2f}%'
     .format(logclf.score(X_train, y_train)*100))
print('The test score: {:.2f}%'
     .format(logclf.score(X_test, y_test)*100))


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


# Sub-Part e: Output the confusion matrix of the test dataset
print('\nPart 4e: Displaying confusion matrix and performance metrics')
print('\nConfusion Matrix:\n')
confmat = confusion_matrix(y_true=y_test, y_pred=y_predict)
print(confmat)

print('\nDisplaying detailed model evaluation\n')
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_predict))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_predict))
print('F1 score: %.3f' % f1_score(y_true=y_test, y_pred=y_predict))


# Sub-Part f: Discuss
print('\nPart 4f: Discussion (in Python code comments)')

"""
The confusion matrix describes how well the logistic model performs on the test data.
It does this by comparing the predicted labels with the actual labels.
In this case there are:
    92 True negatives (TN)
    49 False positives (FP)
    39 False negatives (FN)
    78 True positives (TP)
In total, this amounts to 92+78=170 correct predictions and 49+39=88 incorrect predictions,
which is decent but not great.
Roughly speaking, the model gives a correct prediction on the test data two-thirds of the time.

The precision, recall, and F1 score are metrics which score the performance of the model on the test data.
These metrics range from 0 to 1, with a value closer to 1 indicating better performance.
Each of these scores is computed above in Python, but with explicit formulas given below:
    precision = TP/(TP+FP) = 78/(78+49) = 0.614.
    recall = TP/(FN+TP) = 78/(39+78) = 0.667
    F1 = (2*precision*recall)/(precision+recall) = (2*0.614*0.667)/(0.614+0.667) = 0.639
    
Ultimately, this logistic model isn't very good, although it is better than random.
When looking at the plot of cement vs. water, our two binary classes do not appear to be linearly separable.
Seeing the logistic decision boundary overlaid with the predictions confirms this.
While the plot suggests that a higher cement content is related to a greater compressive strength,
there is too much overlap between the two chosen features for this model to be considered reliable.

"""









from pandas.core.series import Series
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
import pandas as pd
import numpy as np

df_train = pd.read_csv("data/training.csv")
df_test = pd.read_csv("data/test.csv")

X_train, y_train = df_train['tweet'].tolist(), df_train['class'].tolist()
X_test, y_test = df_test['tweet'].tolist(), df_test['class'].tolist()

# Poly Kernel
svm = Pipeline([('pipe', CountVectorizer()),('tfidf', TfidfTransformer()),('svcc', SVC(kernel='poly'))])
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)

acc = metrics.accuracy_score(y_test, prediction)
acc = acc * 100

print("Poly Kernel - Accuracy:", acc, "%")
print(metrics.classification_report(y_test, prediction))

# Linear Kernel
svm = Pipeline([('pipe', CountVectorizer()),('tfidf', TfidfTransformer()),('svcc', SVC(kernel='linear'))])
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)

acc = metrics.accuracy_score(y_test, prediction)
acc = acc * 100

print("Linear Kernel - Accuracy:", acc, "%")
print(metrics.classification_report(y_test, prediction))

# RBF Kernel (Gamma = 10)
svm = Pipeline([('pipe', CountVectorizer()),('tfidf', TfidfTransformer()),('svcc', SVC(kernel='rbf', gamma=10))])
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)

acc = metrics.accuracy_score(y_test, prediction)
acc = acc * 100

print("RBF Kernel (Gamma = 10) - Accuracy:", acc, "%")
print(metrics.classification_report(y_test, prediction))

# RBF Kernel (Gamma = 0.1)
svm = Pipeline([('pipe', CountVectorizer()),('tfidf', TfidfTransformer()),('svcc', SVC(kernel='rbf', gamma=0.1))])
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)

acc = metrics.accuracy_score(y_test, prediction)
acc = acc * 100

print("RBF Kernel (Gamma = 0.1) - Accuracy:", acc, "%")
print(metrics.classification_report(y_test, prediction))

# RBF Kernel (Gamma = 0.7, C = 1)
svm = Pipeline([('pipe', CountVectorizer()),('tfidf', TfidfTransformer()),('svcc', SVC(kernel='rbf', gamma=0.7, C=1))])
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)

acc = metrics.accuracy_score(y_test, prediction)
acc = acc * 100

print("RBF Kernel (Gamma = 0.7, C = 1) - Accuracy:", acc, "%")
print(metrics.classification_report(y_test, prediction))

# Best Results
svm = Pipeline([('pipe', CountVectorizer()),('tfidf', TfidfTransformer()),('svcc', SVC(kernel='rbf', gamma=0.7, C=1.5))])
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)

acc = metrics.accuracy_score(y_test, prediction)
acc = acc * 100

print("RBF Kernel (Gamma = 0.7, C = 1.5) (Best Result) - Accuracy:", acc, "%")
print(metrics.classification_report(y_test, prediction))

df_test["prediction"] = Series(prediction, index=df_test.index)
df_test.to_csv("data/svm_results.csv", index=False)
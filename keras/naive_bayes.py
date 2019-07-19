"""
Created on Thu Oct 18 15:49:39 2018
This code aims to test whether the hypertisan problem can be solved by using naive bayes classfier without any deep learning models.
"""
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json
# import user own package
import data_prepare


# Load the dataset and the labelset
dataset = "../data/small_dataset.txt"
labelset = "../data/small_labelset.txt" 

# set up training size and testing size
train_size = 150000
test_size = 20000

# vocabulary size is set to be 100000
vocabulary_size = 100000

# get articles and labels from the dataset and labelset
articles = data_prepare.load_data(dataset,train_size)
labels = data_prepare.load_labels(labelset,train_size)

# assign index to the words in the article
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(articles)

# turn the words into one-hot vector
X = tokenizer.texts_to_matrix(articles, mode='binary')

clf =  MultinomialNB(alpha=1)
clf.fit(X, labels)

# get the cross validation score
scores = cross_val_score(clf, X, labels, cv=10)
results = {}
results["cross_validation"] = [str(score) for score in scores]


# set up the test data and test label
testdata = "../data/testing_set.txt"
testlabel = "../data/testing_label.txt"

# get articles, labels, and turn words in the test set into one-hot vector
articles = data_prepare.load_data(testdata,test_size)
labels = data_prepare.load_labels(testlabel,test_size)
X_test = tokenizer.texts_to_matrix(articles, mode='binary')

# get the test accuracy and the prediction of the naive bayes model
acc = clf.score(X_test,labels)
results["test accuracy"] = str(float(acc))
y_pred = clf.predict(X_test)
print("confusion matrix",confusion_matrix(y_pred,labels))
print("test accuracy", acc)
print("cross validation accuracy", np.mean(scores))

# writing data to a json file
with open('./output/bayes_result.json', 'w') as outfile:
    json.dump(results, outfile)
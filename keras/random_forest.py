"""
Created on Thu Oct 18 15:49:39 2018
This code aims to test whether the hypertisan problem can be solved by using random forest classfier without any deep learning models.
"""
#classify by Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
import json
# import user own package
import data_prepare

# Load the dataset and the labelset
dataset = "../data/small_dataset.txt"
labelset = "../data/small_labelset.txt"

# set up training size and testing size
data_size = 150000
# vocabulary size is set to be 100000
vocabulary_size = 100000

# get articles and labels from the dataset and labelset
articles = data_prepare.load_data(dataset,data_size)
labels = data_prepare.load_labels(labelset,data_size)

# assign index to the words in the article
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(articles)

# turn the words into one-hot vector
X = tokenizer.texts_to_matrix(articles, mode='binary')

# random forest classifier
my_rdf = RandomForestClassifier(criterion= 'entropy',max_depth=55,n_estimators=260, max_features= 145) #max_depth is the depth of each

#training
my_rdf.fit(X, labels)

# get training accuracy
train_acc = my_rdf.score(X,labels)
results["train accuracy"] = str(float(train_acc))
# get the cross validation score
scores = cross_val_score(my_rdf, X, labels, cv=5)
results = {}
results["cross_validation"] = [str(score) for score in scores]

# set up the test data and test label
testdata = "../data/testing_set.txt"
testlabel = "../data/testing_label.txt"

# get articles, labels, and turn words in the test set into one-hot vector
articles = data_prepare.load_data(testdata,test_size)
labels = data_prepare.load_labels(testlabel,test_size)
X_test = tokenizer.texts_to_matrix(articles, mode='binary')

# get the test accuracy and the prediction of the random forest model
acc = my_rdf.score(X_test,labels)
results["test accuracy"] = str(float(acc))

# get predict label and confusion matrix
y_pred = my_rdf.predict(X_test)
print("confusion matrix",confusion_matrix(y_pred,labels))
print("test accuracy", acc)
print("cross validation accuracy", np.mean(scores))

confusion_matrix = confusion_matrix(y_pred,labels).tolist()
results["test accuracy"] = confusion_matrix
# writing data to a json file
with open('./output/random_forest_result.json', 'w') as outfile:
    json.dump(results, outfile)
importance = my_rdf.feature_importances_  #get importance ranking

#sort feature importance
importance_ind = []
for i in range(importance.shape[0]):
    row = []
    row.append(i)
    imp = importance[i]
    row.append(imp)
    importance_ind.append(row)
importance2 = np.array(importance_ind)
#print importance2
idex=np.lexsort([-1*importance2[:,1]])
importance_sort=importance2[idex,:]

index_word = {v: k for k, v in tokenizer.word_index.items()}
#wordimpsort = []
wordcloud_dct = {}
for i in range(1000):
     ind = importance_sort[i,0]
     word = index_word[int(ind)]
     #wordimpsort.append(word)
     wordcloud_dct[str(word)] = float(importance_sort[i,1])

with open('./output/random_forest_importance.json', 'w') as outfile:
    json.dump(wordcloud_dct, outfile)


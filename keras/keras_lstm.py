'''
Oct. 3
8 GPU, 16G mem, each epoch 2 hours
120 thousands samples as training data, 30 testing samples as testing data
training results: training accuracy 98.55%; test accuracy 89.83%
'''
# Keras
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
import json
# import user own package
import data_prepare

dataset = "../data/small_dataset.txt"
labelset = "../data/small_labelset.txt"

#dataset = "/Users/isrl-elmer/Downloads/NLP/assignment_2/hyperpartisan-nlp/small_dataset.txt"
#labelset = "/Users/isrl-elmer/Downloads/NLP/assignment_2/hyperpartisan-nlp/small_labelset.txt"


# Setting up hyper parameters
data_size = 150000 # The whole dataset is composed of 100,000 articles.
train_size = 120000 # 80,000 articles is used as training data.
test_size = 30000 # 20,000 articles is used as testing data.
batch_size = 100 # Batch size is set to be 100.

embedding_dim = 300 # Dimension of word embedding is 300. Namely, very word is expressed by a vector that has 300 dimensions.
dropout = 0.2 # The dropout rate is set to be 0.2
recurrent_dropout = 0.2 # The recurrent drop rate is set to be 0.2 as well.
vocabulary_size = 100000 # The number of unique words is set to be 100,000. Namely, the 100,000 most frequent word will be used.
sequence_length = 850 # The number of words per article that will be used is set to be 500.
num_epoch = 3 # The number of iteration is set to be 1.



# To get the training data and the corresponding labels, and the testing data and the corresponding labels.
# All the words in both training data and test data have been converted into index.
# The training data is assigned as  X_train, the corresponding label is assigned as y_train,
# and the testing data is assigned as X_test, and the corresponding label is assigned as y_test.
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset, labelset,data_size,vocabulary_size,sequence_length,train_size)

# To get the training loss and accuracy of each step.
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

# To get the testing loss and accuracy of each step.
class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data, logs={}):
        self.test_data = test_data
        self.test_losses = []
        self.test_accs = []

    def on_epoch_end(self, epoch, logs={}):
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose = 0)
        print('\nTesting loss: {}, tacc: {}\n'.format(test_loss, test_acc))
        logs['tloss'] = test_loss
        logs['tacc'] = test_acc
        self.test_losses.append(logs.get('tloss'))
        self.test_accs.append(logs.get('tacc'))


# create the model
model = Sequential() # using sequential model since we are dealing with articles.
model.add(Embedding(vocabulary_size, embedding_dim, input_length=sequence_length)) # Word embedding. To vectorize all the words.
model.add(LSTM(embedding_dim, dropout=dropout, recurrent_dropout=recurrent_dropout)) # Add LSTM model.
model.add(Dense(1, activation='sigmoid')) # Add a hidden layer that uses sigmoid function for classification.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile the model
print(model.summary()) # Print out the model summary

# To get loss and accuracy of each step.
train_history = LossHistory() # Class of training history
test_history = TestCallback((X_test, y_test)) # Class of testing history during model training

# Using training data to train the model
model.fit(X_train, y_train, epochs=num_epoch, batch_size=100, validation_data=(X_test, y_test), callbacks = [train_history, test_history])


# Final evaluation of the model
loss, scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy:", scores[1])

# Loss and accuracy of every batch
train_loss_list = [str(loss) for loss in train_history.losses]
train_acc_list = [str(acc) for acc in train_history.acc]
test_loss_list = [str(tloss) for tloss in test_history.test_losses]
test_acc_list = [str(tacc) for tacc in test_history.test_accs]

# Print out loss and accuracy of every step
print(train_loss_list)
print(train_acc_list)
print(test_loss_list)
print(test_acc_list)


# Writing loss and accuracy into a json file.

with open('train_loss_lstm.json', 'w') as outfile1:
    json.dump(train_loss_list, outfile1)

with open('train_accuracy_lstm.json', 'w') as outfile2:
    json.dump(train_acc_list, outfile2)


with open('test_loss_lstm.json', 'w') as outfile1:
    json.dump(test_loss_list, outfile1)

with open('test_accuracy_lstm.json', 'w') as outfile2:
    json.dump(test_acc_list, outfile2)

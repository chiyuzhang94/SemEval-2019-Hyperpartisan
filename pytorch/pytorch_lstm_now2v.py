import torch
from torch import nn
import torch.autograd as autograd
import json
import pickle
# import user own package
import data_prepare
import train
torch.manual_seed(1)    # reproducible

# Getting data
dataset = "../data/small_dataset.txt"
labelset = "../data/small_labelset.txt"

data_size = 150000 # The whole dataset is composed of 150,000 articles.
train_size = 120000 # 120,000 articles is used as training data.
test_size = 30000 # 30,000 articles is used as testing data.

load_model = False
load_model_path = "./output/pytorch_lstm"

# Hyper Parameters
batch_size = 100  # Batch size is set to be 100.
vocabulary_size = 100000 # The number of unique words is set to be 100000. Namely, the 100000 most frequent word will be used.
sequence_length = 830 # The number of words per article
embedding_dim = 300  # Dimension of word embedding is 300. Namely, very word is expressed by a vector that has 300 dimensions.
num_epoch = 5  # The number of iteration is set to be 5.
hidden_dim = 100 # The number of unit in a hidden layer is set to be 100.
num_layers = 1 # The number of hidden layers is set to be 1.
dropout = 0.2  # The dropout rate is set to be 0.2.
output_size = 2 # The output size is set to be 2.
lr = 0.01           # learning rate is set to be 0.01.

# To define a RNN class
class LSTM(nn.Module):
    def __init__(self,embedding_dim,num_layers,hidden_dim,vocabulary_size,batch_size,dropout):
        super(LSTM, self).__init__() # to inherit the classes of the nn.Module super class
        
        self.embedding_dim = embedding_dim # assign word embedding dimension
        self.hidden_dim = hidden_dim # assign hidden layer dimension
        self.vocab_size = vocabulary_size # assign vocabulary size

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim) # word embedding layer

        self.lstm = nn.LSTM( # LSTM layer
            input_size=embedding_dim, # assign size of each input data
            hidden_size=hidden_dim, # assign hidden layer size
            num_layers = num_layers,  # assing number of layer
            batch_first=True, # batch is the first dimension, followed by embedding_dim and sequence_length
        )
        
        self.hidden2out = nn.Linear(hidden_dim,output_size) # add a hidden layer that has 100 units
        self.softmax = nn.LogSoftmax() # using softmax function
        self.dropout_layer = nn.Dropout(p=dropout) # dropout some units of the hidden layer

    
    def init_hidden(self, batch_size): # initialize the weight of the hidden layer
        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


    def forward(self, x):
        embeds = self.embedding(x) # word embedding layer
        if load_model:
            r_out, (ht, ct) = self.lstm(embeds, None) # If true, get output of the lstm layer without initializing the weights
        else:
            self.hidden = self.init_hidden(batch_size)# If false, intilize the weights
            r_out, (ht, ct) = self.lstm(embeds, self.hidden) # If false, get output of lstm layer using the initial hidden layer
        output = self.dropout_layer(ht[-1]) # get the last ht of lstm and fit into the hidden layer that has been randomly dropped out
        output = self.hidden2out(output) # get the output of the hidden units
        output = self.softmax(output) # output layer using softmax function
        return output
# inherit class LSTM and assign to lstm
lstm = LSTM(embedding_dim,num_layers,hidden_dim,vocabulary_size,batch_size,dropout)
print(lstm)
tokenizer = ''

# if load_model is True, then load the pre-trained model
if load_model:
    lstm.load_state_dict(torch.load(load_model_path))
    with open('./output/lstm_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle) # load tokenzier

optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion

# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset, labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)

# to get the training loss and test accuracy by using the train_no_w2v function of the train.py module
training_loss, training_acc, test_acc = train.train_no_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,lstm,test_size,loss_func,X_test,y_test)

torch.save(lstm.state_dict(), "./output/pytorch_lstm") # save the lstm neural network

with open('./output/lstm_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) # save the model

with open('./output/train_loss_lstm.json', 'w') as outfile1:
    json.dump(training_loss, outfile1) # write training loss results to a json file

with open('./output/test_accuracy_lstm.json', 'w') as outfile2:
    json.dump(test_acc, outfile2) # write test accuracy results to a json file

with open('./output/train_accuracy_lstm.json', 'w') as outfile3:
    json.dump(training_acc, outfile3)  # write training accuracy results to a json file

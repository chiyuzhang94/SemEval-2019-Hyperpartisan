import torch 
import torch.nn as nn
import GPUtil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import math
import torch.nn.functional as F
from time import time
import pickle
import json
import data_prepare
import train
torch.manual_seed(1)    # reproducible
from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

# get data    
dataset = "../data/byarticle_article.txt"
labelset = "../data/byarticle_label.txt"
load_model = False

# use GPU 
cuda_gpu = True

data_size = 645 # the number of articles in dataset
train_size = 580 # train_size articles is used as training data.
test_size = 65 # test_size articles is used as testing data.
num_epoch = 10
# Hyper Parameters
batch_size = 32  # Batch size
sequence_length = 436 # The number of words per article 
embedding_dim = 1024  # Dimension of word embedding is 300. Namely, very word is expressed by a vector that has 300 dimensions.

hidden_dim = 500 # The number of unit in a hidden layer
num_layers = 1 # The number of hidden layers 
dropout = 0.0  # The dropout rate
output_size = 2 # The output size is set to be 2, since we are using the softmax function.
lr = 0.001           # learning rate


# built a lstm structure
class AttentionLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,num_layers,output_size,dropout):
        super(AttentionLSTM,self).__init__() # don't forget to call this!
        
        self.encoder = nn.LSTM(input_size=embedding_dim, # assign size of each input data
                             hidden_size=hidden_dim,
                             num_layers = num_layers,
                           batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.fc1=nn.Linear(hidden_dim,output_size)
        self.hidden2out = nn.Linear(hidden_dim,output_size)
        self.softmax = nn.LogSoftmax()
    
    def attention_net(self, lstm_output, final_state):
        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
        """
        hidden = final_state.permute(1,2,0)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state,attn_weights
  
    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.encoder(x,None)
       
        attn_output,attn_weights = self.attention_net(output, final_hidden_state)
       
        attn_output= self.dropout(attn_output)
        fc_output= self.fc1(attn_output)
        output = self.softmax(fc_output) # output layer using softmax function
        return output

device_ids = GPUtil.getAvailable(limit = 4)
device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")
if cuda_gpu:
    if torch.cuda.device_count() == 1:
        lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout).to(device)
    else:
        torch.backends.cudnn.benchmark = True
        lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout).to(device)
        lstmattn = nn.DataParallel(lstmattn, device_ids=device_ids)
else:
    lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout)

articles = data_prepare.load_data(dataset,data_size)
label_array = data_prepare.load_labels(labelset,data_size)

X_train = articles[0:train_size]
y_train = label_array[0:train_size]

X_test = articles[train_size:]
y_test = label_array[train_size:]

optimizer = torch.optim.Adam(lstmattn.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion

training_loss, training_acc, test_acc = train.train_elmo(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,lstmattn,test_size,loss_func,X_test,y_test,"elmo_lstm",cuda_gpu,elmo)


with open('./output/elmo_lstm/train_loss_lstm.json', 'w') as outfile1:
    json.dump(training_loss, outfile1) # write training loss results to a json file

with open('./output/elmo_lstm/test_accuracy_lstm.json', 'w') as outfile2:
    json.dump(test_acc, outfile2) # write test accuracy results to a json file

with open('./output/lstm/train_accuracy_lstm.json', 'w') as outfile3:
    json.dump(training_acc, outfile3) # write training accuracy results to a json file

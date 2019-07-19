import torch
from torch import nn
import torch.autograd as autograd
import json
import pickle
from time import time
# import user own package
import data_prepare
import train
torch.manual_seed(1)    # reproducible

# Getting data
dataset = "../data/train_dev_article.txt"
labelset = "../data/train_dev_label.txt"

load_model = False
load_model_path = "./output/bilstm/pytorch_bilstm_2"

batch_first = False
w2v_file = "../data/GoogleNews-vectors-negative300.bin" # google news w2v file
w2v_bi = True   # w2v_bi is set to be True since it is a binary file.
# use GPU 
cuda_gpu = True

data_size = 900000 # The whole dataset is composed of data_size articles.
train_size = 800000 # train_size articles is used as training data.
test_size = 100000 # test_size articles is used as testing data.
# Hyper Parameters
batch_size = 50# Batch size 
vocabulary_size = 40000 # The number of unique words in vocabulary
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 6  # The number of iteration

hidden_dim = 200 # The number of unit in a hidden layer
num_layers = 1  # The number of hidden layers 

dropout = 0.0  # The dropout rate 
output_size = 2 # The output size 
lr = 0.01        # learning rate 

# To define a RNN class
class LSTM(nn.Module):
    def __init__(self,embedding_dim,num_layers,hidden_dim,batch_size,dropout,cuda_gpu):
        super(LSTM, self).__init__() # to inherit the LSTM class of the nn.Module super class
        
        self.hidden_dim = hidden_dim # assign hidden layer dimension

        self.lstm = nn.LSTM( # LSTM layer
            input_size=embedding_dim, # assign size of each input data
            hidden_size=hidden_dim, # assign hidden layer size
            num_layers = num_layers,  # assing number of layer
            bidirectional=True, 
            #batch_first=True,
        )
        
        self.hidden2out = nn.Linear(hidden_dim*2,output_size) # add a hidden layer
        self.softmax = nn.LogSoftmax() # using softmax function
        self.dropout_layer = nn.Dropout(p=dropout) # dropout some units of the hidden layer

    
    def init_hidden(self, batch_size): # initialize the weight of the hidden layer
        if cuda_gpu:
            return(autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).cuda(),
                   autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).cuda())
        else:
            return(autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)),
            autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)))

    def forward(self, x):

        if load_model:
            r_out, (ht, ct) = self.lstm(x, None) # If true, get output of the lstm layer without using the initial hidden layer
        else:
            self.hidden = self.init_hidden(batch_size)
            r_out, (ht, ct) = self.lstm(x, self.hidden) # If false, get output of lstm layer using the initial hidden layer

        output = self.dropout_layer(r_out[-1]) # get the output of lstm and fit into the hidden layer that has been randomly dropped out
        output = self.hidden2out(output) # get the output of the hidden units
        output = self.softmax(output) # output layer using softmax function
        return output
# inherit class LSTM and assign to lstm
if cuda_gpu:
    bilstm = LSTM(embedding_dim,num_layers,hidden_dim,batch_size,dropout,cuda_gpu).cuda()
else:
    bilstm = LSTM(embedding_dim,num_layers,hidden_dim,batch_size,dropout,cuda_gpu)
print(bilstm)

tokenizer = ''

# if load_model is True, then load the pre-trained model
if load_model:
    lstm.load_state_dict(torch.load(load_model_path))
    with open('./output/bilstm/bilstm_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

optimizer = torch.optim.Adam(bilstm.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion

# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset,labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)

with open('./output/bilstm/bilstm_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) # save the model
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)

start = time()
# to get the training loss and test accuracy by using the train.with_w2v function of the train.py module
training_loss, training_acc, test_acc = train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,bilstm,test_size,loss_func,X_test,y_test,"bilstm",batch_first,cuda_gpu)

end = time()
print("time",end - start)

with open('./output/bilstm/train_loss_bilstm.json', 'w') as outfile1:
    json.dump(training_loss, outfile1)  # write training loss results to a json file

with open('./output/bilstm/test_accuracy_bilstm.json', 'w') as outfile2:
    json.dump(test_acc, outfile2) # write test accuracy results to a json file

with open('./output/bilstm/train_accuracy_bilstm.json', 'w') as outfile3:
    json.dump(training_acc, outfile3)  # write training accuracy results to a json file


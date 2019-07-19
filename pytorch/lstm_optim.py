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
dataset = "../data/train_dev_article.txt"
labelset = "../data/train_dev_label.txt"

w2v_file = "../data/GoogleNews-vectors-negative300.bin" # google news w2v file
w2v_bi = True   # w2v_bi is set to be True since it is a binary file.

# use GPU 
cuda_gpu = True

batch_first = True
data_size = 110000 # the number of articles in dataset.
train_size = 100000 # 120,000 articles is used as training data.
test_size = 10000 # 30,000 articles is used as testing data.
# Hyper Parameters
batch_size = 50 # Batch size
vocabulary_size = 40000 # The number of unique words 
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 3  # The number of iteration 
num_layers = 1   # The number of hidden layers
output_size = 2  # The output size

##################################################################################################################################################################
# define the parameters that need to optimize
op_parameters = []

var_dic = {}
var_dic['var_name'] = 'lr'
var_dic['var_ini'] = 0.01
var_dic['var_range'] = [1,0.1,0.01,0.001,0.0001]
op_parameters.append(var_dic)

var_dic = {}
var_dic['var_name'] = "hidden_dim"
var_dic['var_ini'] = 100
var_dic['var_range'] = range(100,501,100)
op_parameters.append(var_dic)

var_dic = {}
var_dic['var_name'] = "dropout"
var_dic['var_ini'] = 0.1
var_dic['var_range'] = [0.0,0.1,0.3,0.5,0.7,0.9]
op_parameters.append(var_dic)
##################################################################################################################################################################


# To define a RNN class
class LSTM(nn.Module):
    def __init__(self,embedding_dim,num_layers,hidden_dim,batch_size,dropout,cuda_gpu):
        super(LSTM, self).__init__() # to inherit the LSTM class of the nn.Module super class
        
        self.hidden_dim = hidden_dim # assign hidden layer dimension

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
        if cuda_gpu:
            return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).cuda(),
            autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).cuda())
        else:
            return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
            autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, x):
        if load_model:
            r_out, (ht, ct) = self.lstm(x, None) # If true, get output of the lstm layer without using the initial hidden layer
        else:
            self.hidden = self.init_hidden(batch_size)
            r_out, (ht, ct) = self.lstm(x, self.hidden) # If false, get output of lstm layer using the initial hidden layer
        output = self.dropout_layer(ht[-1]) # get the output of lstm and fit into the hidden layer that has been randomly dropped out
        output = self.hidden2out(output) # get the output of the hidden units
        output = self.softmax(output) # output layer using softmax function
        return output

tokenizer = ''
# don't change, optimization don't need to load model
load_model = False
# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset,labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)

# initialize values
for i, item in enumerate(op_parameters):
    exec("%s = %f" % (item['var_name'],item['var_ini']))

results = {}

for item in op_parameters:
    acc = 0.0
    best = 0.0
    process = {}
    for value in item['var_range']:
        exec("%s = %f" % (item['var_name'],value))
        if cuda_gpu:
            lstm = LSTM(int(embedding_dim),int(num_layers),int(hidden_dim),int(batch_size),dropout,cuda_gpu).cuda()
        else:
            lstm = LSTM(int(embedding_dim),int(num_layers),int(hidden_dim),int(batch_size),dropout,cuda_gpu)
        
        optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)   # define a optimizer for backpropagation
        loss_func = nn.CrossEntropyLoss()   # define loss funtion

        training_loss,training_acc, test_acc = train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,lstm,test_size,loss_func,X_test,y_test,"lstm",batch_first,cuda_gpu)
        
        process[value] = [test_acc[-1],training_acc[-1]]
        if acc < test_acc[-1]:
            acc = test_acc[-1]
            best = value

    print("the best %s dimension is %f. Accuracy is %f", (item['var_name'],best,acc))
    
    results[item['var_name']+'_best'] = best
    results[item['var_name']+'_process'] = process
    exec("%s = %f" % (item['var_name'],best))

# store optimization results
with open('./output/lstm/optimize_lstm.json', 'w') as outfile2:
    json.dump(results, outfile2)


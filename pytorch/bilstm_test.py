import torch
import torch.nn as nn
import torch.autograd as autograd
import json
import pickle
# import user own package
import data_prepare
import test
torch.manual_seed(1)    # reproducible

# get data    
test_set = "../data/test_article.txt"
test_label = "../data/test_label.txt"
load_model = True
load_model_path = "./output/bilstm/pytorch_bilstm_3"
w2v_file = "../data/GoogleNews-vectors-negative300.bin" 
w2v_bi = True

# use GPU 
cuda_gpu = False

batch_first = False
data_size = 100 # the number of articles in dataset
# Hyper Parameters
batch_size = 20  # Batch size
vocabulary_size = 40000 # The number of unique words is set to be 100000. Namely, the 100000 most frequent word will be used.
sequence_length = 436 # The number of words per article 
embedding_dim = 300  # Dimension of word embedding is 300. Namely, very word is expressed by a vector that has 300 dimensions.

hidden_dim = 200 # The number of unit in a hidden layer
num_layers = 1 # The number of hidden layers 
dropout = 0.0 # The dropout rate
output_size = 2 # The output size is set to be 2, since we are using the softmax function.
lr = 0.001           # learning rate
# don't change, test always needs to load model
load_model = True

# To define a bilstm class
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
    bilstm.load_state_dict(torch.load(load_model_path))
else:
    bilstm = LSTM(embedding_dim,num_layers,hidden_dim,batch_size,dropout,cuda_gpu)
    bilstm.load_state_dict(torch.load(load_model_path, map_location='cpu'))
print(bilstm)

with open('./output/bilstm/bilstm_tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer) # use the google w2v vector as the embedding layer
# use test function in test package, get test accurcy, precision, recall and f1 score
acc,output_dic = test.test_with_w2v(test_set,test_label,data_size,vocabulary_size,sequence_length,load_model,tokenizer,batch_size,embedding_dim,embedding_matrix,bilstm,batch_first)
print("accuracy",acc)
# store results
with open('./output/bilstm/test_output_bilstm.json', 'w') as outfile2:
    json.dump(output_dic, outfile2)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from time import time
import pickle
import json
# import user own package
import data_prepare
import test
torch.manual_seed(1)    # reproducible

# get data    
test_set = "../data/test_article.txt"
test_label = "../data/test_label.txt"
load_model = True
load_model_path = "./output/lstmattn/pytorch_lstmattn_7"
w2v_file = "../data/GoogleNews-vectors-negative300.bin" 
w2v_bi = True

# use GPU 
cuda_gpu = False

batch_first = True
data_size = 100 # the number of articles in dataset
# Hyper Parameters
batch_size = 100000  # Batch size
vocabulary_size = 40000 # The number of unique words is set to be 100000. Namely, the 100000 most frequent word will be used.
sequence_length = 436 # The number of words per article 
embedding_dim = 300  # Dimension of word embedding is 300. Namely, very word is expressed by a vector that has 300 dimensions.

hidden_dim = 500 # The number of unit in a hidden layer
num_layers = 1 # The number of hidden layers 
dropout = 0.0  # The dropout rate
output_size = 2 # The output size is set to be 2, since we are using the softmax function.
lr = 0.001           # learning rate
# don't change, test always needs to load model
load_model = True
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
        return new_hidden_state
  
    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.encoder(x,None)
       
        attn_output = self.attention_net(output, final_hidden_state)
       
        attn_output= self.dropout(attn_output)
        fc_output= self.fc1(attn_output)
        output = self.softmax(fc_output) # output layer using softmax function
        return output

if cuda_gpu:
    lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout).cuda()
    lstmattn.load_state_dict(torch.load(load_model_path))
else:
    lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout)
    lstmattn.load_state_dict(torch.load(load_model_path, map_location='cpu'))
print(lstmattn)


with open('./output/lstmattn/lstmattn_tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer) # use the google w2v vector as the embedding layer
# use test function in test package, get test accurcy, precision, recall and f1 score
acc,output_dic = test.test_with_w2v(test_set,test_label,data_size,vocabulary_size,sequence_length,load_model,tokenizer,batch_size,embedding_dim,embedding_matrix,lstmattn,batch_first)
print("accuracy",acc)
# store results
with open('./output/lstmattn/test_output_lstmattn.json', 'w') as outfile2:
    json.dump(output_dic, outfile2)
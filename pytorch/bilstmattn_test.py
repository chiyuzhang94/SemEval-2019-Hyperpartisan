import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from time import time
import pickle
import json
import GPUtil
import numpy as np
from collections import OrderedDict
# import user own package
import data_prepare
import test
torch.manual_seed(1)    # reproducible

# get data    
test_set = "../data/test_article.txt"
test_label = "../data/test_label.txt"
load_model = True
load_model_path = "./output/bilstmattn_test/pytorch_bilstmattn_6.pt"
w2v_file = "../data/GoogleNews-vectors-negative300.bin" 
w2v_bi = True

# use GPU 
cuda_gpu = False

batch_first = True
data_size = 100 # the number of articles in dataset
# Hyper Parameters
batch_size = 100  # Batch size
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
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1=nn.Linear(hidden_dim*2,output_size)
        self.hidden2out = nn.Linear(hidden_dim,output_size)
        self.softmax = nn.LogSoftmax()
    
    def attention_net(self, lstm_output, final_state):
        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
        """
        hidden = final_state
        lstm_output = lstm_output.permute(1,0,2)

        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
  
    def forward(self, x):
        x = torch.transpose(x,0,1)
        output, (final_hidden_state, final_cell_state) = self.encoder(x,None)
       
        attn_output = self.attention_net(output, output[-1])
        attn_output= self.dropout(attn_output)
        fc_output= self.fc1(attn_output)
        output = self.softmax(fc_output) # output layer using softmax function
        return output

##########################################################
lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout)
print("model done")
if torch.cuda.is_available():
    device_ids = GPUtil.getAvailable(limit = 4)
device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")
##########################################################
tokenizer = ''
if load_model:
    with open('./output/bilstmattn_test/bilstmattn_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    lstmattn_checkpoint = torch.load(load_model_path,map_location='cpu')['state_dict']
    
    new_state_dict = OrderedDict()
    for k, v in lstmattn_checkpoint.items():
        if "module" in k:
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    
    lstmattn.load_state_dict(new_state_dict)
##########################################################
if cuda_gpu:
    print(device_ids)
    if torch.cuda.device_count() == 1:
        lstmattn = lstmattn.cuda()
    else:
        torch.backends.cudnn.benchmark = True
        lstmattn = lstmattn.cuda(device_ids[0])
        lstmattn = nn.DataParallel(lstmattn, device_ids=device_ids)
else:
    lstmattn = lstmattn
print(lstmattn)

# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer) # use the google w2v vector as the embedding layer
# use test function in test package, get test accurcy, precision, recall and f1 score
acc,output_dic = test.test_with_w2v(test_set,test_label,data_size,vocabulary_size,sequence_length,load_model,tokenizer,batch_size,embedding_dim,embedding_matrix,lstmattn,batch_first)
print("accuracy",acc)
# store results
with open('./output/bilstmattn_test/test_output_lstmattn.json', 'w') as outfile2:
    json.dump(output_dic, outfile2)
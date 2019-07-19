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
# get data    
model_name = 'bert_lstmattn'
dataset = "../data/train_dev_article.txt"
labelset = "../data/train_dev_label.txt"
load_model = False
load_model_path = "./output/"+model_name+"/pytorch_bert_lstmattn_2"
# use GPU 
cuda_gpu = True

data_size = 90
train_size = 80 
test_size = 100 
num_epoch = 5
# Hyper Parameters
batch_size = 50 
sequence_length = 436 
embedding_dim = 1024

hidden_dim = 500 
num_layers = 1 
dropout = 0.0  
output_size = 2 
lr = 0.001      


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
        lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout).cuda()
        if load_model:
            lstmattn.load_state_dict(torch.load(load_model_path))
    else:
        torch.backends.cudnn.benchmark = True
        lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout).cuda(device_ids[0])
        if load_model:
            lstmattn.load_state_dict(torch.load(load_model_path))
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

training_loss, training_acc, test_acc = train.train_with_bert(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,lstmattn,test_size,loss_func,X_test,y_test,model_name,cuda_gpu)


with open('./output/'+model_name+'/train_loss_lstm.json', 'w') as outfile1:
    json.dump(training_loss, outfile1) # write training loss results to a json file

with open('./output/'+model_name+'/test_accuracy_lstm.json', 'w') as outfile2:
    json.dump(test_acc, outfile2) # write test accuracy results to a json file

with open('./output/'+model_name+'/train_accuracy_lstm.json', 'w') as outfile3:
    json.dump(training_acc, outfile3) # write training accuracy results to a json file

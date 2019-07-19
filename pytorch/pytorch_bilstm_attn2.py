import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from time import time
import pickle
import json
from collections import OrderedDict
import GPUtil
import numpy as np
# import user own package
import data_prepare
import train

torch.manual_seed(1)    # reproducible

# Getting data
auto_dataset = "../data/train_dev_article.txt"
auto_labelset = "../data/train_dev_label.txt"
auto_test_set = "../data/test_article.txt"
auto_test_label = "../data/test_label.txt"

load_model = False
load_model_path = "./output/bilstmattn_auto2/pytorch_bilstmattn_0.pt"

model_name = "bilstmattn"
outpath = './output/bilstmattn_auto2/'

batch_first = True
w2v_file = "../data/GoogleNews-vectors-negative300.bin" 
w2v_bi = True  # The w2v_bi variable is set to be True
# use GPU 
cuda_gpu = True

train_size = 980000 # train_size articles is used as training data.
test_size = 20000 # test_size articles is used as testing data.
# Hyper Parameters
batch_size = 32 # Batch size 
vocabulary_size = 40000 # The number of unique words in vocabulary
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 15  # The number of iteration

hidden_dim = 500
num_layers = 1

output_size = 2 # The output size is set to be 2.
lr = 0.001
dropout = 0.0

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

tokenizer = ''
if load_model:
    lstmattn_checkpoint = torch.load(load_model_path,map_location='cpu')["state_dict"]
    
    new_state_dict = OrderedDict()
    for k, v in lstmattn_checkpoint.items():
        if "module" in k:
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    
    lstmattn.load_state_dict(new_state_dict)

##########################################################
if torch.cuda.is_available():
    device_ids = GPUtil.getAvailable(limit = 4)
device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")
##########################################################
print("model done")
if cuda_gpu:
    if torch.cuda.device_count() == 1:
        lstmattn = lstmattn.to(device)
    else:
        print("more gpus")
        device_ids = GPUtil.getAvailable(limit = 4)
        torch.backends.cudnn.benchmark = True
        lstmattn = lstmattn.to(device)
        lstmattn = nn.DataParallel(lstmattn, device_ids=device_ids)
else:
    lstmattn = lstmattn
print(lstmattn)

# to get training data and test data
all_train_art = []
all_train_lab = []

auto_valid_art = []
auto_valid_label = []

art_test = data_prepare.load_data(auto_test_set,test_size)
y_test = data_prepare.load_labels(auto_test_label,test_size).tolist()
auto_valid_art.extend(art_test)
auto_valid_label.extend(y_test)


art_auto = data_prepare.load_data(auto_dataset,train_size)
y_auto = data_prepare.load_labels(auto_labelset,train_size).tolist()

all_train_art.extend(art_auto)
all_train_lab.extend(y_auto)

##########################################################
with open('./output/bilstmattn_auto2/bilstmattn_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
load_model = True

X_train,tokenizer = data_prepare.tokenize(all_train_art,vocabulary_size,sequence_length,load_model,tokenizer)
y_train = np.asarray(all_train_lab)

X_test,tokenizer = data_prepare.tokenize(auto_valid_art,vocabulary_size,sequence_length,True,tokenizer)
y_test= np.asarray(auto_valid_label)
load_model = False
##########################################################
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)

with open(outpath+'bilstmattn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) # save the model

optimizer = torch.optim.Adam(lstmattn.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion
start = time()

training_loss, training_acc, test_acc =  train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,lstmattn,test_size,loss_func,X_test,y_test,model_name,batch_first,cuda_gpu,outpath)

end = time()
print("time",end - start)

with open(outpath+'train_loss_bilstmattn.json', 'w') as outfile1:
    json.dump(training_loss, outfile1)  # write training loss results to a json file

with open(outpath+'test_accuracy_bilstmattn.json', 'w') as outfile2:
    json.dump(test_acc, outfile2) # write test accuracy results to a json file

with open(outpath+'train_accuracy_bilstmattn.json', 'w') as outfile3:
    json.dump(training_acc, outfile3)  # write training accuracy results to a json file


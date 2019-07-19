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
# Getting data
auto_dataset = "../data/train_dev_article.txt"
auto_labelset = "../data/train_dev_label.txt"
auto_test_set = "../data/test_article.txt"
auto_test_label = "../data/test_label.txt"
##########################################################
outpath = "./output/bilstmattn_final/"
model_name = "bilstmattn"

load_model = True
load_model_path = "./output/pytorch_bilstmattn_0.pt"

batch_first = True
w2v_file = "../data/GoogleNews-vectors-negative300.bin" # google news w2v file
w2v_bi = True   # w2v_bi is set to be True since it is a binary file.
# use GPU 
cuda_gpu = True

auto_data_size = 90
auto_testdata_size = 10
# Hyper Parameters
batch_size = 32# Batch size 
vocabulary_size = 40000 # The number of unique words in vocabulary
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 1 # The number of iteration

hidden_dim = 500 # The number of unit in a hidden layer
num_layers = 1  # The number of hidden layers 

dropout = 0.0  # The dropout rate 
output_size = 2 # The output size 
lr = 0.001        # learning rate 

##########################################################
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


lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout)
print("model done")
if torch.cuda.is_available():
    device_ids = GPUtil.getAvailable(limit = 5)
device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")
##########################################################
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
if cuda_gpu:
    print(device_ids)
    if torch.cuda.device_count() == 1:
        lstmattn = lstmattn.to(device)
    else:
        print("more gpus")
        torch.backends.cudnn.benchmark = True
        lstmattn = lstmattn.to(device)
        lstmattn = nn.DataParallel(lstmattn, device_ids=device_ids)
else:
    lstmattn = lstmattn
print(lstmattn)

optimizer = torch.optim.Adam(lstmattn.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion

# to get training data
##########################################################
all_train_art = []
all_train_lab = []

art_test = data_prepare.load_data(auto_test_set,auto_testdata_size)
y_test = data_prepare.load_labels(auto_test_label,auto_testdata_size).tolist()
all_train_art.extend(art_test)
all_train_lab.extend(y_test)

#################################################################
art_auto = data_prepare.load_data(auto_dataset,auto_data_size)
y_auto = data_prepare.load_labels(auto_labelset,auto_data_size).tolist()

all_train_art.extend(art_auto)
all_train_lab.extend(y_auto)

#################################################################
with open('./output/bilstmattn_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
load_model = True
train_size = len(all_train_art)
X_train,tokenizer = data_prepare.tokenize(all_train_art,vocabulary_size,sequence_length,load_model,tokenizer)
y_train = np.asarray(all_train_lab)
load_model = False
##########################################################

with open(outpath+'bilstm_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) # save the model
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)

start = time()
# to get the training loss and test accuracy by using the train.with_w2v function of the train.py module
training_loss, training_acc = train.train_w2v_final(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,lstmattn,loss_func,model_name,batch_first,cuda_gpu,outpath)

end = time()
print("time",end - start)

with open(outpath+'train_loss_bilstm.json', 'w') as outfile1:
    json.dump(training_loss, outfile1)  # write training loss results to a json file


with open(outpath+'train_accuracy_bilstmattn.json', 'w') as outfile3:
    json.dump(training_acc, outfile3)  # write training accuracy results to a json file


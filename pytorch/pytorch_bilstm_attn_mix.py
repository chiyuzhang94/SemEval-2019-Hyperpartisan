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
import train

torch.manual_seed(1)    # reproducible
##########################################################
# Getting data
auto_dataset = "../data/train_dev_article.txt"
auto_labelset = "../data/train_dev_label.txt"
auto_test_set = "../data/test_article.txt"
auto_test_label = "../data/test_label.txt"

manual_dataset = "../data/byarticle_article.txt"
manual_labelset = "../data/byarticle_label.txt"
##########################################################
outpath = "./output/bilstmattn_mix/"
model_name = "bilstmattn"

load_model = False
load_model_path = "./output/bilstmattn_mix/pytorch_bilstmattn_4.pt"
##########################################################
batch_first = True
w2v_file = "../data/GoogleNews-vectors-negative300.bin" 
w2v_bi = True  # The w2v_bi variable is set to be True
# use GPU 
cuda_gpu = True
device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")
##########################################################
auto_data_size = 900000
man_data_size = 600
auto_testdata_size = 100000

auto_valid_size = 50000
man_valid_size = 60
##########################################################
# Hyper Parameters
batch_size = 16 # Batch size 
vocabulary_size = 40000 # The number of unique words in vocabulary
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 10 # The number of iteration

hidden_dim = 500
num_layers = 1

output_size = 2 # The output size is set to be 2.
lr = 0.001
dropout = 0.0
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
##########################################################
lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout)
print("model done")
if torch.cuda.is_available():
    device_ids = GPUtil.getAvailable(limit = 4)
device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")
##########################################################
tokenizer = ''
if load_model:
    # with open(outpath+'bilstmattn_tokenizer.pickle', 'rb') as handle:
    #     tokenizer = pickle.load(handle)
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
##########################################################
all_train_art = []
all_train_lab = []

auto_valid_art = []
auto_valid_label = []

man_valid_art = []
man_valid_label = []
# to get training data and test data
############################################################
art_test = data_prepare.load_data(auto_test_set,auto_testdata_size)
y_test = data_prepare.load_labels(auto_test_label,auto_testdata_size).tolist()
all_train_art.extend(art_test)
all_train_lab.extend(y_test)
#################################################################
art_auto = data_prepare.load_data(auto_dataset,auto_data_size)
y_auto = data_prepare.load_labels(auto_labelset,auto_data_size).tolist()

art_auto_train = art_auto[0:len(art_auto)-auto_valid_size]
y_auto_train = y_auto[0:len(art_auto)-auto_valid_size]

art_auto_valid = art_auto[len(art_auto)-auto_valid_size:]
y_auto_valid = y_auto[len(art_auto)-auto_valid_size:]

all_train_art.extend(art_auto_train)
all_train_lab.extend(y_auto_train)

auto_valid_art.extend(art_auto_valid)
auto_valid_label.extend(y_auto_valid)

###########################################################
art_manual = data_prepare.load_data(manual_dataset,man_data_size)
y_manual = data_prepare.load_labels(manual_labelset,man_data_size).tolist()

art_man_train = art_manual[0:len(art_manual)-man_valid_size]
y_man_train = y_manual[0:len(art_manual)-man_valid_size]

art_man_valid = art_manual[len(art_manual)-man_valid_size:]
y_man_valid = y_manual[len(art_manual)-man_valid_size:]

all_train_art.extend(art_man_train)
all_train_lab.extend(y_man_train)

man_valid_art.extend(art_man_valid)
man_valid_label.extend(y_man_valid)
##########################################################
with open(outpath+'bilstmattn_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
load_model = True
train_size = len(all_train_art)
X_train,tokenizer = data_prepare.tokenize(all_train_art,vocabulary_size,sequence_length,load_model,tokenizer)
y_train = np.asarray(all_train_lab)

X_test_auto,tokenizer = data_prepare.tokenize(auto_valid_art,vocabulary_size,sequence_length,True,tokenizer)
y_test_auto = np.asarray(auto_valid_label)

X_test_man,tokenizer = data_prepare.tokenize(man_valid_art,vocabulary_size,sequence_length,True,tokenizer)
y_test_man = np.asarray(man_valid_label)
load_model = False
##########################################################
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)
##########################################################
with open(outpath+'bilstmattn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) # save the model
##########################################################
optimizer = torch.optim.Adam(lstmattn.parameters(), lr=lr)   # define a optimizer for backpropagation
if load_model:
    optimizer.load_state_dict(torch.load(load_model_path)['optimizer'])

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
loss_func = nn.CrossEntropyLoss()   # define loss funtion
start = time()
##########################################################
training_loss, auto_acc, man_acc =  train.train_w2v_mix(num_epoch,train_size,batch_size,optimizer,X_train,y_train,X_test_auto,y_test_auto,X_test_man,y_test_man,auto_valid_size,man_valid_size,sequence_length,embedding_dim,embedding_matrix,lstmattn,loss_func,model_name,batch_first,cuda_gpu,scheduler,outpath)
##########################################################
end = time()
print("time",end - start)

with open(outpath+'train_loss_lstmattn.json', 'w') as outfile1:
    json.dump(training_loss, outfile1)  # write training loss results to a json file

with open(outpath+'auto_accuracy_lstmattn.json', 'w') as outfile2:
    json.dump(auto_acc, outfile2) # write test accuracy results to a json file

with open(outpath+'man_accuracy_lstmattn.json', 'w') as outfile3:
    json.dump(man_acc, outfile3)  # write training accuracy results to a json file

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
import train

torch.manual_seed(1)    # reproducible

# Getting data
dataset = "../data/train_dev_article.txt"
labelset = "../data/train_dev_label.txt"

load_model = False
load_model_path = "./output/lstmattn/pytorch_lstmattn_2"

batch_first = True
w2v_file = "../data/GoogleNews-vectors-negative300.bin" 
w2v_bi = True  # The w2v_bi variable is set to be True
# use GPU 
cuda_gpu = True

data_size = 900000 # The whole dataset is composed of data_size articles.
train_size = 800000# train_size articles is used as training data.
test_size = 100000 # test_size articles is used as testing data.
# Hyper Parameters
batch_size = 100 # Batch size 
vocabulary_size = 40000 # The number of unique words in vocabulary
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 8  # The number of iteration

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
        print(attn_weights.shape)
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
else:
    lstmattn = AttentionLSTM(embedding_dim, hidden_dim,num_layers,output_size,dropout)
print(lstmattn)

tokenizer = ''
# if load_model is True, then load the pre-trained model
if load_model:
    lstmattn.load_state_dict(torch.load(load_model_path))
    with open('./output/lstmattn/lstmattn_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)


# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset,labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)

with open('./output/lstmattn/lstmattn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) # save the model

optimizer = torch.optim.Adam(lstmattn.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion
start = time()

training_loss, training_acc, test_acc = train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,lstmattn,test_size,loss_func,X_test,y_test,"lstmattn",batch_first,cuda_gpu)

end = time()
print("time",end - start)

with open('./output/lstmattn/train_loss_lstmattn.json', 'w') as outfile1:
    json.dump(training_loss, outfile1)  # write training loss results to a json file


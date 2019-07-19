import torch
import torch.nn.functional as F
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
load_model_path = "../output/clstm/pytorch_clstm_2"

batch_first = True
w2v_file = "../data/GoogleNews-vectors-negative300.bin" 
w2v_bi = True  # The w2v_bi variable is set to be True

load_model = False
load_model_path = "./output/pytorch_clstm"
# use GPU 
cuda_gpu = True

data_size = 70 # The whole dataset is composed of data_size articles.
train_size = 60# train_size articles is used as training data.
test_size = 10 # test_size articles is used as testing data.
# Hyper Parameters
batch_size = 10# Batch size 
vocabulary_size = 40000 # The number of unique words in vocabulary
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 3  # The number of iteration

lstm_hidden_dim = 200
lstm_num_layers = 1
kernel_num = 100
kernel_sizes = [2,3,4]

output_size = 2
dropout = 0.2
lr = 0.01

class CLSTM(nn.Module):
    
    def __init__(self, lstm_hidden_dim,lstm_num_layers,embedding_dim,output_size,kernel_num,kernel_sizes,dropout,cuda_gpu):
        super(CLSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        D = embedding_dim
        C = output_size
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        # CNN
        KK = []
        for K in Ks:
            KK.append( K + 1 if K % 2 == 0 else K)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K//2, 0)) for K in KK ]) 
        
        # LSTM
        self.lstm = nn.LSTM(input_size=len(KK)*Co, # assign size of each input data
                             hidden_size=self.hidden_dim,
                             num_layers = self.num_layers,
                           batch_first = True)

        # linear
        self.hidden2label = nn.Linear(self.hidden_dim,output_size)
        self.softmax = nn.LogSoftmax() 
        # dropout
        self.dropout_layer = nn.Dropout(p=dropout) 
    def init_hidden(self, batch_size): # initialize the weight of the hidden layer
        if cuda_gpu:
            return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).cuda(),
                   autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).cuda())
        else:
            return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
            autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, x):
        # CNN
        cnn_x = x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)

        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1, 2)
        # LSTM
        self.hidden = self.init_hidden(batch_size)
             
        lstm_out, (ht, ct) = self.lstm(cnn_x,self.hidden)
        lstm_out = self.dropout_layer(ht[-1])
        
        # linear
        
        cnn_lstm_out = self.hidden2label(lstm_out)
        cnn_lstm_out = self.softmax(cnn_lstm_out)
        # output
        logit = cnn_lstm_out

        return logit

if cuda_gpu:    
    clstm = CLSTM(lstm_hidden_dim,lstm_num_layers,embedding_dim,output_size,kernel_num,kernel_sizes,dropout,cuda_gpu).cuda()
else:
    clstm = CLSTM(lstm_hidden_dim,lstm_num_layers,embedding_dim,output_size,kernel_num,kernel_sizes,dropout,cuda_gpu)
print(clstm)

tokenizer = ''
# if load_model is True, then load the pre-trained model
if load_model:
    attlstm.load_state_dict(torch.load(load_model_path))
    with open('./output/clstm/clstm_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

optimizer = torch.optim.Adam(clstm.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion

# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset,labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)

with open('./output/clstm/clstm_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) # save the model
    
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)
start = time()

training_loss, training_acc, test_acc = train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,clstm,test_size,loss_func,X_test,y_test,"clstm",batch_first,cuda_gpu)

end = time()
print("time",end - start)

with open('./output/clstm/train_loss_clstm.json', 'w') as outfile1:
    json.dump(training_loss, outfile1)  # write training loss results to a json file

with open('./output/clstm/test_accuracy_clstm.json', 'w') as outfile2:
    json.dump(test_acc, outfile2) # write test accuracy results to a json file

with open('./output/clstm/train_accuracy_clstm.json', 'w') as outfile3:
    json.dump(training_acc, outfile3)  # write training accuracy results to a json file

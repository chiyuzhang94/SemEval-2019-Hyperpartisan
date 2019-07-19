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
import test

torch.manual_seed(1)    # reproducible
     
test_set = "../data/test_article.txtt"
test_labels = "../data/test_label.txt"

load_model_path = "./output/clstm/pytorch_clstm_01"
w2v_file = "../data/GoogleNews-vectors-negative300.bin" 
w2v_bi = True  # The w2v_bi variable is set to be True
batch_first = True
data_size = 100 # the number of articles in dataset
# use GPU 
cuda_gpu = False
# Hyper Parameters
batch_size = 100000 # Batch size is set to be 100.
vocabulary_size = 40000 # The number of unique words
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding

lstm_hidden_dim = 200
lstm_num_layers = 1
kernel_num = 100
kernel_sizes = [2,3,4]

dropout = 0.1  # The dropout rate 
output_size = 2 # The output size 
lr = 0.01        # learning rate

# don't change, test always needs to load model
load_model = True
# built a cnn structure
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
        self.lstm = nn.LSTM(input_size=len(KK) * Co, # assign size of each input data
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
    clstm.load_state_dict(torch.load(load_model_path))
else:
    clstm = CLSTM(lstm_hidden_dim,lstm_num_layers,embedding_dim,output_size,kernel_num,kernel_sizes,dropout,cuda_gpu)
    clstm.load_state_dict(torch.load(load_model_path, map_location='cpu'))
print(clstm)

with open('./output/clstm/clstm_tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer) # use the google w2v vector as the embedding layer
# use test function in test package, get test accurcy, precision, recall and f1 score
acc, output_dic = test.test_with_w2v(test_set,test_labels,data_size,vocabulary_size,sequence_length,load_model,tokenizer,batch_size,embedding_dim,embedding_matrix,cnn,batch_first)
print("accuracy",acc)
# store results
with open('./output/clstm/test_output_clstm.json', 'w') as outfile2:
    json.dump(output_dic, outfile2)

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
load_model_path = "./output/rcnn/pytorch_rcnn_3"
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
dropout = 0.3 # The dropout rate
output_size = 2 # The output size is set to be 2, since we are using the softmax function.
lr = 0.001           # learning rate
# don't change, test always needs to load model
load_model = True


class RCNN(nn.Module):
    def __init__(self, batch_size, output_size, hidden_dim, embedding_dim, dropout,cuda_gpu):
        super(RCNN, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=self.dropout, bidirectional=True)
        self.W2 = nn.Linear(2*hidden_dim+embedding_dim, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax() # using softmax function
    
    def init_hidden(self, batch_size): # initialize the weight of the hidden layer
        if cuda_gpu:
            return(autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).cuda(),
            autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).cuda())
        else:
            return(autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)),
            autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)))
    def forward(self, x):
        if load_model:
            output, (final_hidden_state, final_cell_state) = self.lstm(x, None) # If true, get output of the lstm layer without using the initial hidden layer
        else:
            self.hidden = self.init_hidden(batch_size)
            output, (final_hidden_state, final_cell_state) = self.lstm(x, self.hidden) # If false, get output of lstm layer using the initial hidden layer

        final_encoding = torch.cat((output, x), 2).permute(1, 0, 2)
        y = self.W2(final_encoding) # y.size() = (batch_size, num_sequences, hidden_dim)
        y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_dim, num_sequences)
        y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_dim, 1)
        y = y.squeeze(2)
        y = self.hidden2out(y)
        y = self.softmax(y) # output layer using softmax function
        
        return y
# inherit class RCNN  and assign to lstm
if cuda_gpu:
    rcnn = RCNN(embedding_dim,num_layers,hidden_dim,batch_size,dropout,cuda_gpu).cuda()
    rcnn.load_state_dict(torch.load(load_model_path))
else:
    rcnn = RCNN(embedding_dim,num_layers,hidden_dim,batch_size,dropout,cuda_gpu)
    rcnn.load_state_dict(torch.load(load_model_path, map_location='cpu'))
print(rcnn)

with open('./output/rcnn/rcnn_tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer) # use the google w2v vector as the embedding layer
# use test function in test package, get test accurcy, precision, recall and f1 score
acc,output_dic = test.test_with_w2v(test_set,test_label,data_size,vocabulary_size,sequence_length,load_model,tokenizer,batch_size,embedding_dim,embedding_matrix,rcnn,batch_first)
print("accuracy",acc)
# store results
with open('./output/rcnn/test_output_rcnn.json', 'w') as outfile2:
    json.dump(output_dic, outfile2)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import json
import pickle
# import user own package
import data_prepare
import test

torch.manual_seed(1)    # reproducible
     
test_set = "../data/test_article.txtt"
test_labels = "../data/test_label.txt"

load_model_path = "./output/cnn/pytorch_cnn_7"
w2v_file = "../data/GoogleNews-vectors-negative300.bin" 
w2v_bi = True  # The w2v_bi variable is set to be True
batch_first = True
data_size = 100 # the number of articles in dataset
# use GPU 
cuda_gpu = False
# Hyper Parameters
batch_size = 100 # Batch size is set to be 100.
vocabulary_size = 40000 # The number of unique words
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding

kernel_sizes = [4,5,6] # region size of kernel
kernel_num = 200 # the number of filters
dropout = 0.1  # The dropout rate 
output_size = 2 # The output size 
lr = 0.01        # learning rate

# don't change, test always needs to load model
load_model = True
# built a cnn structure
class CNN_Text(nn.Module):
    
    def __init__(self, embedding_dim,output_size, kernel_num,kernel_sizes,vocabulary_size,dropout):
        super(CNN_Text, self).__init__()
        D = embedding_dim # set up variables
        C = output_size
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks]) # convolution with filter
        self.dropout = nn.Dropout(dropout) # a dropout layer
        self.fc1 = nn.Linear(len(Ks)*Co, C) # to get features that map for each region size

    def conv_and_pool(self, x, conv):  # a max pooling function
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W) filter the input
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # pool the max value of the filtered data
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch size, number of channel is one, word_sequence, embeeding_dim) change the shape into a tensor
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] # [(batch size, number of channel, word_sequence), ...]*len(Ks) apply the filter to the input and get the output as a tensor
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks) # to get the maximum value of filtered tensor
        x = torch.cat(x, 1) # reshape the matrix
        x = self.dropout(x)  # (N, len(Ks)*Co) add drop layer
        logit = self.fc1(x)  # (N, C) to get the fc1 using a fully connected layer
        return logit
# inherit class CNN_Text and assign to cnn
cnn = CNN_Text(embedding_dim,output_size, kernel_num,kernel_sizes,vocabulary_size,dropout)
print(cnn)

cnn.load_state_dict(torch.load(load_model_path))
with open('./output/cnn/cnn_tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer) # use the google w2v vector as the embedding layer
# use test function in test package, get test accurcy, precision, recall and f1 score
acc, output_dic = test.test_with_w2v(test_set,test_labels,data_size,vocabulary_size,sequence_length,load_model,tokenizer,batch_size,embedding_dim,embedding_matrix,cnn,batch_first)
print("accuracy",acc)
# store results
with open('./output/cnn/test_output_cnn.json', 'w') as outfile2:
    json.dump(output_dic, outfile2)

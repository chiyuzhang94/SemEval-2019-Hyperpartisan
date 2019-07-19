import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import json
import pickle
from time import time
# import user own package
import data_prepare
import train

torch.manual_seed(1)    # reproducible
     
dataset = "../data/train_dev_articles.txt"
labelset = "../data/train_dev_label.txt"

load_model =  True
batch_first = True
w2v_file = "../data/word2vec_org" 
w2v_bi = False  # The w2v_bi variable
# use GPU 
cuda_gpu = True

data_size = 70 # The whole dataset is composed of data_size articles.
train_size = 60 # train_size articles is used as training data.
test_size = 10 # test_size articles is used as testing data.
# Hyper Parameters
batch_size = 10 # Batch size 
vocabulary_size = 40000 # The number of unique words in vocabulary
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 3  # The number of iteration

kernel_sizes = [4,5,6] # filter size as 3, 4, and 5
kernel_num = 190 # the number of filters

dropout = 0.1  # The dropout rate is set to be 0.2.
output_size = 2 # The output size is set to be 2.
lr = 0.01        # learning rate is set to be 0.01.

# To define a CNN class
class CNN_Text(nn.Module):
    
    def __init__(self, embedding_dim,output_size, kernel_num,kernel_sizes,vocabulary_size,dropou):
        super(CNN_Text, self).__init__()
        D = embedding_dim # set up variables
        C = output_size
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks]) # convolution with filter
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (Ks1, D))
        self.conv14 = nn.Conv2d(Ci, Co, (Ks2, D))
        self.conv15 = nn.Conv2d(Ci, Co, (Ks3, D))
        '''
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

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co) add drop layer
        logit = self.fc1(x)  # (N, C) to get the fc1 using a fully connected layer
        return logit
# inherit class CNN_Text and assign to cnn
if cuda_gpu:
    cnn = CNN_Text(embedding_dim,output_size, kernel_num,kernel_sizes,vocabulary_size,dropout).cuda()
else:
    cnn = CNN_Text(embedding_dim,output_size, kernel_num,kernel_sizes,vocabulary_size,dropout)
print(cnn)

tokenizer = ''

# if load_model is True, then load the pre-trained model
if load_model:
    with open('./output/uni_w2v_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset, labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)

# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer) # use the google w2v vector as the embedding layer

optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion

start = time()

# to get the training loss and test accuracy by using the train_with_w2v function of the train.py module
training_loss, training_acc, test_acc = train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,cnn,test_size,loss_func,X_test,y_test,"cnnds",batch_first,cuda_gpu)

end = time()
print("time",end - start)

with open('./output/train_loss_cnn_dsw2v.json', 'w') as outfile1:
    json.dump(training_loss, outfile1)  # write training loss results to a json file

with open('./output/test_accuracy_cnn_dsw2v.json', 'w') as outfile2:
    json.dump(test_acc, outfile2) # write test accuracy results to a json file

with open('./output/train_accuracy_cnn_dsw2v.json', 'w') as outfile3:
    json.dump(training_acc, outfile3)  # write training accuracy results to a json file

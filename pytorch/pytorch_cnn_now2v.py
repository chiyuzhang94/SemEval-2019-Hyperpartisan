import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import json
import pickle
# import user own package
import data_prepare
import train
torch.manual_seed(1)    # reproducible

# Getting data
dataset = "../data/small_dataset.txt"
labelset = "../data/small_labelset.txt"

load_model = False
load_model_path = "../output/pytorch_cnn"

data_size = 150000 # The whole dataset is composed of 150,000 articles.
train_size = 120000 # 120,000 articles is used as training data.
test_size = 30000 # 30,000 articles is used as testing data.

# Hyper Parameters
batch_size = 100 # Batch size is set to be 100.
vocabulary_size = 100000 # The number of unique words is set to be 100000. Namely, the 100000 most frequent word will be used.
sequence_length = 830 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding is 300. Namely, very word is expressed by a vector that has 300 dimensions.
num_epoch = 5  # The number of iteration is set to be 5.

kernel_sizes = [3,4,5] # filter size as 3, 4, and 5
kernel_num = 100 # the number of filters
dropout = 0.2  # The dropout rate is set to be 0.2.
output_size = 2 # The output size is set to be 2.
lr = 0.01        # learning rate is set to be 0.01.

# To define a CNN class
class CNN_Text(nn.Module):
    def __init__(self, embedding_dim,output_size, kernel_num,kernel_sizes,vocabulary_size,dropout):
        super(CNN_Text, self).__init__()
#         self.args = args
        D = embedding_dim # set up variables
        C = output_size
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes

        self.embed = nn.Embedding(vocabulary_size, D) # word embedding layer
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks]) # convolution with filter
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (Ks1, D))
        self.conv14 = nn.Conv2d(Ci, Co, (Ks2, D))
        self.conv15 = nn.Conv2d(Ci, Co, (Ks3, D))
        '''
        self.dropout = nn.Dropout(dropout) # a dropout layer
        self.fc1 = nn.Linear(len(Ks)*Co, C) # to get features that map for each region size

    def conv_and_pool(self, x, conv): # a max pooling function
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W) filter the input
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # pool the max value of the filtered data
        return x

    def forward(self, x):
        x = self.embed(x)  # (batch size, word_sequence, embedding_dim) word embedding
        x = x.unsqueeze(1)  # (batch size, number of channel is one, word_sequence, embeeding_dim) change the shape into a tensor
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch size, number of channel, word_sequence), ...]*len(Ks)  apply the filter to the input and get the output as a tensor
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks) # to get the maximum value of filtered tensor
        x = torch.cat(x, 1) # reshape the matrix
        x = self.dropout(x)  # (N, len(Ks)*Co) add drop layer
        logit = self.fc1(x)  # (N, C) to get the fc1 using a fully connected layer
        return logit
# inherit class CNN_Text and assign to cnn
cnn = CNN_Text(embedding_dim,output_size, kernel_num,kernel_sizes,vocabulary_size,dropout)
print(cnn)
tokenizer = ''

# if load_model is True, then load the pre-trained model
if load_model:
    cnn.load_state_dict(torch.load(load_model_path))
    with open('./output/cnn_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset, labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)

optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion

# to get the training loss and test accuracy by using the train_no_w2v function of the train.py module
training_loss, training_acc, test_acc = train.train_no_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,cnn,test_size,loss_func,X_test,y_test)

torch.save(cnn.state_dict(), "./output/pytorch_cnn") # save the cnn neural network

with open('./output/cnn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)  # save the model

with open('./output/train_loss_cnn.json', 'w') as outfile1:
    json.dump(training_loss, outfile1) # write training loss results to a json file

with open('./output/test_accuracy_cnn.json', 'w') as outfile2:
    json.dump(test_acc, outfile2)  # write test accuracy results to a json file

with open('./output/train_accuracy_cnn.json', 'w') as outfile3:
    json.dump(training_acc, outfile3)  # write training accuracy results to a json file

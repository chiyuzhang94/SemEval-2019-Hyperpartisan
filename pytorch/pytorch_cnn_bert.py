import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import json
import pickle
import GPUtil
from time import time
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

manual_dataset = "../data/byarticle_article.txt"
manual_labelset = "../data/byarticle_label.txt"

outpath = "./output/bert_cnn_mix/"
model_name = "bert_cnn"

load_model = False
load_model_path = "./output/cnn/pytorch_cnn_2"

batch_first = True
# use GPU 
cuda_gpu = True

auto_data_size = 6000
man_data_size = 600
auto_testdata_size = 10

auto_valid_size = 600
man_valid_size = 40

# Hyper Parameters
batch_size = 8 # Batch size 
vocabulary_size = 40000 # The number of unique words in vocabulary
sequence_length = 436 # The number of words per article 
embedding_dim = 1024 # Dimension of word embedding

num_epoch = 3  # The number of iteration

kernel_sizes = [4,5,6] # filter size as 3, 4, and 5
kernel_num = 190 # the number of filters

dropout = 0.1  # The dropout rate is set to be 0.2.
output_size = 2 # The output size is set to be 2.
lr = 0.01        # learning rate is set to be 0.01.

device_ids = GPUtil.getAvailable(limit = 4)
device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")

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

cnn = CNN_Text(embedding_dim,output_size, kernel_num,kernel_sizes,vocabulary_size,dropout)
# inherit class CNN_Text and assign to cnn
if torch.cuda.device_count() <= 1:
    cnn.to(device)
else:
    print("use more gpus")
    torch.backends.cudnn.benchmark = True
    cnn = cnn.to(device)
    cnn = nn.DataParallel(cnn, device_ids=device_ids)

if load_model:
    cnn_checkpoint = torch.load(load_model_path,map_location='cpu')['state_dict']
    
    new_state_dict = OrderedDict()
    for k, v in cnn_checkpoint.items():
        if "module" in k:
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v  
    cnn.load_state_dict(new_state_dict)

all_train_art = []
all_train_lab = []

auto_valid_art = []
auto_valid_label = []

man_valid_art = []
man_valid_label = []
# to get training data and test data
###########################################################
art_test = data_prepare.load_data(auto_test_set,auto_testdata_size)
y_test = data_prepare.load_labels(auto_test_label,auto_testdata_size).tolist()
all_train_art.extend(art_test)
all_train_lab.extend(y_test)
###########################################################
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
###########################################################
all_train_art.extend(art_man_train)
all_train_lab.extend(y_man_train)

man_valid_art.extend(art_man_valid)
man_valid_label.extend(y_man_valid)
###########################################################
data_coun = open(outpath+"label_ds.txt","w")
train_size = len(all_train_art)
X_train = all_train_art
y_train = np.asarray(all_train_lab)
data_coun.write(str(np.sum(y_train))+"y_train")

X_test_auto = auto_valid_art
y_test_auto = np.asarray(auto_valid_label)
data_coun.write(str(np.sum(y_test_auto))+"y_test_auto")

X_test_man = man_valid_art
y_test_man = np.asarray(man_valid_label)
data_coun.write(str(np.sum(y_test_man))+"y_test_man")
data_coun.close()

optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion


# to get the training loss and test accuracy by using the train_with_w2v function of the train.py module
training_loss, auto_acc, man_acc = train.train_bert_mix(num_epoch,train_size,batch_size,optimizer,X_train,y_train,X_test_auto,y_test_auto,X_test_man,y_test_man,auto_valid_size,man_valid_size,sequence_length,embedding_dim,cnn,loss_func,model_name,batch_first,cuda_gpu,outpath)

with open(outpath+'train_loss_lstmattn.json', 'w') as outfile1:
    json.dump(training_loss, outfile1)  # write training loss results to a json file

with open(outpath+'auto_accuracy_lstmattn.json', 'w') as outfile2:
    json.dump(auto_acc, outfile2) # write test accuracy results to a json file

with open(outpath+'man_accuracy_lstmattn.json', 'w') as outfile3:
    json.dump(man_acc, outfile3)  # write training accuracy results to a json file

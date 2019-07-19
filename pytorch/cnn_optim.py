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
dataset = "../data/train_dev_article.txt"
labelset = "../data/train_dev_label.txt"

w2v_file = "../data/GoogleNews-vectors-negative300.bin" # google news w2v file
w2v_bi = True   # w2v_bi is set to be True since it is a binary file.

# use gpu
cuda_gpu = True

batch_first = True
data_size = 110 # the number of articles in dataset.
train_size = 100 # 120,000 articles is used as training data.
test_size = 10 # 30,000 articles is used as testing data.

# Hyper Parameters
batch_size = 5 # Batch size
vocabulary_size = 40000 # The number of unique words 
num_epoch = 3 # The number of iteration
sequence_length = 436 # The number of words per article 
embedding_dim = 300  # Dimension of word embedding
output_size = 2 # The output size is set to be 2, since we are using the softmax function.

##################################################################################################################################################################
# define the parameters that need to optimize
op_parameters = []

var_dic = {}
var_dic['var_name'] = 'lr'
var_dic['var_ini'] = 0.01
var_dic['var_range'] = [1,0.1,0.01,0.001,0.0001]
op_parameters.append(var_dic)

var_dic = {}
var_dic['var_name'] = 'kernel_sizes'
var_dic['var_ini'] = '3,4,5'
var_dic['var_range'] = ['2,3,4','3,4,5','4,5,6','5,6,7','6,7,8','7,8,9']
op_parameters.append(var_dic)

var_dic = {}
var_dic['var_name'] = 'kernel_num'
var_dic['var_ini'] = 100
var_dic['var_range'] = range(10,300,30)
op_parameters.append(var_dic)

var_dic = {}
var_dic['var_name'] = "dropout"
var_dic['var_ini'] = 0.1
var_dic['var_range'] = [0.0,0.1,0.3,0.5,0.7,0.9]
op_parameters.append(var_dic)
##################################################################################################################################################################

# To define a CNN class
class CNN_Text(nn.Module):
    def __init__(self, embedding_dim,output_size, kernel_num,kernel_sizes,vocabulary_size,dropout):
        super(CNN_Text, self).__init__()
        D = embedding_dim
        C = output_size
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        # create a model list include all filters of different kernel
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (Ks1, D))
        self.conv14 = nn.Conv2d(Ci, Co, (Ks2, D))
        self.conv15 = nn.Conv2d(Ci, Co, (Ks3, D))
        '''
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

tokenizer = ''
# don't change, optimization don't need to load model
load_model = False

# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset,labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)

# initialize values
for i, item in enumerate(op_parameters):
    if "," in str(item['var_ini']):
        list_value = [int(x) for x in item['var_ini'].split(",")]
        exec("%s = %s" % (item['var_name'],"list_value"))
    else:    
        exec("%s = %f" % (item['var_name'],item['var_ini']))

results = {}
print(kernel_sizes)
for item in op_parameters:
    acc = 0.0
    best = 0.0
    process = {}
    for value in item['var_range']:
        if "," in str(value):
            list_value = [int(x) for x in value.split(",")]
            exec("%s = %s" % (item['var_name'],"list_value"))
        else:
            exec("%s = %f" % (item['var_name'],value))
        print(kernel_sizes)  
        if cuda_gpu:    
            cnn = CNN_Text(int(embedding_dim),int(output_size),int(kernel_num),kernel_sizes,dropout,cuda_gpu).cuda()
        else:
            cnn = CNN_Text(int(embedding_dim),int(output_size),int(kernel_num),kernel_sizes,dropout,cuda_gpu)
        
        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)   # define a optimizer for backpropagation
        loss_func = nn.CrossEntropyLoss()   # define loss funtion

        training_loss,training_acc, test_acc = train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,cnn,test_size,loss_func,X_test,y_test,"cnn",batch_first,cuda_gpu)
        
        process[value] = [test_acc[-1],training_acc[-1]]
        if acc < test_acc[-1]:
            acc = test_acc[-1]
            best = value

    print("the best %s dimension is %f. Accuracy is %f", (item['var_name'],best,acc))
    
    results[item['var_name']+'_best'] = best
    results[item['var_name']+'_process'] = process
    if "," in str(item['var_ini']):
        exec("%s = %s" % (item['var_name'],best))
    else:
        exec("%s = %f" % (item['var_name'],best))

# store optimization results
with open('./output/cnn/optimize_cnn.json', 'w') as outfile2:
    json.dump(results, outfile2)






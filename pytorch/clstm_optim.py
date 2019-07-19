import torch
import torch.nn.functional as F
import torch
from torch import nn
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

# use GPU 
cuda_gpu = True

batch_first = True
data_size = 110000 # the number of articles in dataset.
train_size = 100000 # 120,000 articles is used as training data.
test_size = 10000 # 30,000 articles is used as testing data.
# Hyper Parameters
batch_size = 50 # Batch size
vocabulary_size = 40000 # The number of unique words 
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 3  # The number of iteration
output_size = 2
lstm_num_layers = 1
##################################################################################################################################################################
# define the parameters that need to optimize
op_parameters = []

var_dic = {}
var_dic['var_name'] = 'lr'
var_dic['var_ini'] = 0.01
var_dic['var_range'] = [1,0.1,0.01,0.001,0.0001]
op_parameters.append(var_dic)

var_dic = {}
var_dic['var_name'] = "lstm_hidden_dim"
var_dic['var_ini'] = 100
var_dic['var_range'] = range(100,601,100)
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
        self.lstm = nn.LSTM(input_size=len(KK)*Ks, # assign size of each input data
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
load_model = False
tokenizer = ''

# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset,labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)

# initialize values
for i, item in enumerate(op_parameters):
    if "," in str(item['var_ini']):
        list_value = [int(x) for x in item['var_ini'].split(",")]
        exec("%s = %s" % (item['var_name'],"list_value"))
    else:    
        exec("%s = %f" % (item['var_name'],item['var_ini']))

results = {}

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
            
        if cuda_gpu:    
            clstm = CLSTM(int(lstm_hidden_dim),int(lstm_num_layers),int(embedding_dim),int(output_size),int(kernel_num),kernel_sizes,dropout,cuda_gpu).cuda()
        else:
            clstm = CLSTM(int(lstm_hidden_dim),int(lstm_num_layers),int(embedding_dim),int(output_size),int(kernel_num),kernel_sizes,dropout,cuda_gpu)
        
        optimizer = torch.optim.Adam(clstm.parameters(), lr=lr)   # define a optimizer for backpropagation
        loss_func = nn.CrossEntropyLoss()   # define loss funtion

        training_loss,training_acc, test_acc = train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,clstm,test_size,loss_func,X_test,y_test,"clstm",batch_first,cuda_gpu)
        
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
with open('./output/clstm/optimize_clstm.json', 'w') as outfile2:
    json.dump(results, outfile2)







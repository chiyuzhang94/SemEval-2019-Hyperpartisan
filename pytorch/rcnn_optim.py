import torch
from torch import nn
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

# use GPU 
cuda_gpu = True

batch_first = False
data_size = 55000 # the number of articles in dataset.
train_size = 50000 # the articles is used as training data.
test_size = 5000 # the articles is used as testing data.
# Hyper Parameters
batch_size = 5 # Batch size
vocabulary_size = 40000 # The number of unique words 
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 3  # The number of iteration
output_size = 2 # The output size is set to be 2.
num_layers = 1  # The number of hidden layers is set to be 1.

##################################################################################################################################################################
# define the parameters that need to optimize
op_parameters = []

var_dic = {}
var_dic['var_name'] = 'lr'
var_dic['var_ini'] = 0.01
var_dic['var_range'] = [1,0.1,0.01,0.001,0.0001]
op_parameters.append(var_dic)

var_dic = {}
var_dic['var_name'] = "hidden_dim"
var_dic['var_ini'] = 100
var_dic['var_range'] = range(100,601,100)
op_parameters.append(var_dic)

var_dic = {}
var_dic['var_name'] = "dropout"
var_dic['var_ini'] = 0.1
var_dic['var_range'] = [0.0,0.1,0.3,0.5,0.7,0.9]
op_parameters.append(var_dic)
##################################################################################################################################################################

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

tokenizer = ''
load_model = False
# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset,labelset,data_size,sequence_length,vocabulary_size,train_size,load_model,tokenizer)
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)
print("load embedding")
# initialize values
for i, item in enumerate(op_parameters):
    exec("%s = %f" % (item['var_name'],item['var_ini']))

results = {}

for item in op_parameters:
	acc = 0.0
	best = 0.0
	process = {}
	for value in item['var_range']:
		print(item['var_name'],value)
		exec("%s = %f" % (item['var_name'],value))
		if cuda_gpu:
			rcnn = RCNN(batch_size, output_size, int(hidden_dim), embedding_dim,dropout,cuda_gpu).cuda()
		else:
			rcnn = RCNN(batch_size, output_size, int(hidden_dim), embedding_dim,dropout,cuda_gpu)
		optimizer = torch.optim.Adam(rcnn.parameters(), lr=lr)   # define a optimizer for backpropagation
		loss_func = nn.CrossEntropyLoss()   # define loss funtion

		training_loss,training_acc, test_acc = train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,rcnn,test_size,loss_func,X_test,y_test,"rcnn",batch_first,cuda_gpu)

		process[value] = [test_acc[-1],training_acc[-1]]
		if acc < test_acc[-1]:
			acc = test_acc[-1]
			best = value

	print("the best %s dimension is %f. Accuracy is %f", (item['var_name'],best,acc))

	results[item['var_name']+'_best'] = best
	results[item['var_name']+'_process'] = process
	exec("%s = %f" % (item['var_name'],best))

# store optimization results
with open('./output/rcnn/optimize_rcnn.json', 'w') as outfile2:
    json.dump(results, outfile2)

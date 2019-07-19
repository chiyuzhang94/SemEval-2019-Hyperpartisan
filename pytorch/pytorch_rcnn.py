import torch
from torch import nn
import torch.nn.functional as F
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
load_model_path = "./output/rcnn/pytorch_rcnn_2"

w2v_file = "../data/GoogleNews-vectors-negative300.bin" # google news w2v file
w2v_bi = True   # w2v_bi is set to be True since it is a binary file.

# use GPU 
cuda_gpu = True

batch_first = False
data_size = 70 # The whole dataset is composed of data_size articles.
train_size = 60# train_size articles is used as training data.
test_size = 10 # test_size articles is used as testing data.
# Hyper Parameters
batch_size = 5# Batch size 
vocabulary_size = 40000 # The number of unique words in vocabulary
sequence_length = 436 # The number of words per article 
embedding_dim = 300 # Dimension of word embedding
num_epoch = 3  # The number of iteration

hidden_dim = 200 # The number of unit in a hidden layer is set to be 200.
num_layers = 1  # The number of hidden layers is set to be 1.

dropout = 0.3  # The dropout rate is set to be 0.2.
output_size = 2 # The output size is set to be 2.
lr = 0.001        # learning rate is set to be 0.01.

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

# inherit class LSTM and assign to lstm
if cuda_gpu:
    rcnn = RCNN(batch_size, output_size, hidden_dim, embedding_dim,dropout,cuda_gpu).cuda()
else:
    rcnn = RCNN(batch_size, output_size, hidden_dim, embedding_dim,dropout,cuda_gpu)
print(rcnn)

tokenizer = ''

# if load_model is True, then load the pre-trained model
if load_model:
    rcnn.load_state_dict(torch.load(load_model_path))
    with open('./output/rcnn/rcnn_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

optimizer = torch.optim.Adam(rcnn.parameters(), lr=lr)   # define a optimizer for backpropagation
loss_func = nn.CrossEntropyLoss()   # define loss funtion

# to get training data and test data
X_train, y_train, X_test, y_test, tokenizer = data_prepare.data_ready(dataset,labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer)

with open('./output/rcnn/rcnn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) # save the model
# using google news w2v as word embedding model
embedding_matrix = data_prepare.load_w2v(w2v_file, w2v_bi, vocabulary_size, embedding_dim,tokenizer)
start = time()

# to get the training loss and test accuracy by using the train.with_w2v function of the train.py module
training_loss, training_acc, test_acc = train.train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,rcnn,test_size,loss_func,X_test,y_test,"rcnn",batch_first,cuda_gpu)

end = time()
print("time",end - start)

with open('./output/rcnn/train_loss_rcnn.json', 'w') as outfile1:
    json.dump(training_loss, outfile1) # write training loss results to a json file

with open('./output/rcnn/test_accuracy_rcnn.json', 'w') as outfile2:
    json.dump(test_acc, outfile2) # write test accuracy results to a json file

with open('./output/rcnn/train_accuracy_rcnn.json', 'w') as outfile3:
    json.dump(training_acc, outfile3) # write training accuracy results to a json file
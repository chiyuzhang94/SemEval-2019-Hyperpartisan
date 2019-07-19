# the following codes are packaged as a py file that can be import in other python scripts. 
# this user own package is named as "test"
import torch
import numpy as np
from time import time
# import user own package
import data_prepare
# define a function for model test with pre-trained word2vec
def test_with_w2v(test_set,test_labels,data_size,vocabulary_size,sequence_length,load_model,tokenizer,batch_size,embedding_dim,embedding_matrix,model,batch_first):
    # load data and labels
    articles = data_prepare.load_data(test_set,data_size)
    labels = data_prepare.load_labels(test_labels,data_size)
    # tokenize and transform to matrix
    X, tokenizer = data_prepare.tokenize(articles,vocabulary_size,sequence_length,load_model,tokenizer)
    pred_labels = [] 
    correct = 0
    total = 0
    acc = 0
    # predict label for test data by given model
    for i in range(0,data_size,batch_size):# mini batch process
        batch_start = i
        batch_end = i+batch_size
        if batch_end > data_size-1:
            batch_end = data_size

        if batch_first: 
            input_x = data_prepare.trans2input_batch_first(X[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
        else:
            input_x = data_prepare.trans2input_batch_second(X[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors

        b_x = torch.from_numpy(input_x).float()   # reshape x to (batch, time_step, input_size)
    
        test_output = model(b_x)  # model output
        del b_x
        y_true = np.asarray(labels[batch_start:batch_end])
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
        pred_labels.extend(pred_y.tolist())
        total += y_true.shape[0] # the total number of article in the test data
        correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true
    
    acc = 100.00 * float(correct) / float(total) # get accuracy
    ## calculate presicion, recall and F1 score
    t0 = 0.0
    t1 = 0.0
    f0 = 0.0
    f1 = 0.0
    for i in range(len(pred_labels)):
        true_label = labels[i]
        predict = pred_labels[i]
        if predict == 1 and true_label == 1:
            t1 += 1
        elif predict == 0 and true_label == 0:
            t0 += 1
        elif true_label == 1 and predict == 0:
            f1 += 1
        elif true_label == 0 and predict == 1:
            f0 += 1
    
    precision0 = t0/(t0+f0)
    recall0 = t0/(t0+f1)
    f1_score0 = 2*((precision0*recall0)/(precision0+recall0))
    
    precision1 = t1/(t1+f1)
    recall1 = t1/(t1+f0)
    f1_score1 = 2*((precision1*recall1)/(precision1+recall1))
    # store all results into a dictionary
    output_dic = {}
    output_dic["t0"] = t0 
    output_dic["t1"] = t1
    output_dic["f0"] = f0
    output_dic["f1"] = f1
    output_dic["precision0"] = precision0
    output_dic["recall0"] = recall0
    output_dic["f1_score0"] = f1_score0
    output_dic["precision1"] = precision1
    output_dic["recall1"] = recall1
    output_dic["f1_score1"] = f1_score1
    output_dic["test accuracy"] = acc
    
    return acc,output_dic
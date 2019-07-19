# the following codes aree packaged as a py file that can be import in other python scripts. 
# this user own package is named as "train"
import torch
import numpy as np
from time import time
# import user own package
import data_prepare
# define a function for model training without pre-trained word2vec
def train_no_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,model,test_size,loss_func,X_test,y_test):
    correct = 0 # set up variables
    total = 0
    step = 0
    training_loss = []
    test_acc = []
    for epoch in range(num_epoch):
        for i in range(0,train_size,batch_size):  # gives batch data 
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)
            
            step += 1    # clear gradients for this training step
            optimizer.zero_grad() # set the gradiant vector as zero
            
            b_x = torch.from_numpy(X_train[batch_start:batch_end,:]).long()   # reshape training x to (batch, time_step, input_size)
            b_y = torch.from_numpy(y_train[batch_start:batch_end]).long()     # reshape training y to (batch, time_step, input_size)
            
            output = model(b_x)               # lstm output
            loss = loss_func(output, b_y)   # cross entropy loss
            
            print("STEP",str(step),"lost is",loss.item())
            training_loss.append(loss.item())       
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradient descent
        
        for j in range(0,test_size,batch_size): # test every batch
            b_x = torch.from_numpy(X_test[j:j+batch_size,:]).long()   # reshape testing x to (batch, time_step, input_size)
            
            test_output = model(b_x).float()   # convert b_x to float
            
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            y_true = torch.from_numpy(y_test[j:j+batch_size]).long()  # reshape testing y to (batch, time_step, input_size)
            total += y_true.size(0) # the total number of article in the test data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true
        
        acc = 100.00 * float(correct) / float(total) # get accuracy
        print('Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        test_acc.append(acc)

    return training_loss,test_acc
# define a function for model training with pre-trained word2vec
def train_with_w2v(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,model,test_size,loss_func,X_test,y_test,model_name,batch_first,cuda_gpu,outpath):
    import GPUtil
    device_ids = GPUtil.getAvailable(limit = 4)
    device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    acc = 0
    step = 0
    training_loss = []
    training_acc = []
    test_acc = []
    print("training")
    for epoch in range(num_epoch):
        outfile = open(outpath+"/pytorch_"+model_name+".txt",'a')
        # train the model and back-propagate by batch process
        for i in range(0,train_size,batch_size):   # gives batch data
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)

            step += 1
            optimizer.zero_grad()
            if batch_first: 
                input_x = data_prepare.trans2input_batch_first(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            else:
                input_x = data_prepare.trans2input_batch_second(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            
            if cuda_gpu:
                b_x = torch.from_numpy(input_x).float().to(device)  # reshape training x to (batch, time_step, input_size)
                b_y = torch.from_numpy(y_train[batch_start:batch_end]).long().to(device) # reshape training y to (batch, time_step, input_size)
            else:
                b_x = torch.from_numpy(input_x).float()   # reshape training x to (batch, time_step, input_size)
                b_y = torch.from_numpy(y_train[batch_start:batch_end]).long() # reshape training y to (batch, time_step, input_size)
            del input_x
            output = model(b_x)               # lstm output
            del b_x
            loss = loss_func(output, b_y)   # cross entropy loss
            if step%100 == 0:
                print("STEP",str(step),"lost is",loss.item())
            
            training_loss.append(loss.item())       
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
        # get train accuracy for each epoch
        for i in range(0,train_size,batch_size):
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)
            if batch_first: 
                input_x = data_prepare.trans2input_batch_first(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            else:
                input_x = data_prepare.trans2input_batch_second(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            
            if cuda_gpu:
                b_x = torch.from_numpy(input_x).float().to(device)    # reshape training x to (batch, time_step, input_size)
            else:
                b_x = torch.from_numpy(input_x).float()   # reshape training x to (batch, time_step, input_size)
            del input_x
            train_output = model(b_x)  # lstm output
            del b_x

            if cuda_gpu:
                pred_y = torch.max(train_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(train_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            
            y_true = np.asarray(y_train[batch_start:batch_end]) # reshape training y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the training data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true
        
        acc = 100.00 * (float(correct) / float(total)) # get accuracy
        print('Training Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        training_acc.append(acc)
        train_acc = acc
        # get test accuracy for each epoch
        correct = 0
        total = 0
        acc = 0
        for i in range(0,test_size,batch_size):
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_test)-1:
                batch_end = len(X_test)

            if batch_first: 
                input_x = data_prepare.trans2input_batch_first(X_test[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            else:
                input_x = data_prepare.trans2input_batch_second(X_test[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            
            if cuda_gpu:
                b_x = torch.from_numpy(input_x).float().to(device)   # reshape training x to (batch, time_step, input_size)
            else:
                b_x = torch.from_numpy(input_x).float()   # reshape training x to (batch, time_step, input_size)
            del input_x
            test_output = model(b_x)  # lstm output
            del b_x

            if cuda_gpu:
                pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze() 
            
            y_true = np.asarray(y_test[batch_start:batch_end]) # reshape testing y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the test data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true
        
        acc = 100.00 * (float(correct) / float(total))# get accuracy
        print('Testing Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        test_acc.append(acc)
        testing_acc = acc
        # save model afer each epoch
        if torch.cuda.device_count() <= 1:
            state_dict_model = model.state_dict()
        else:
            state_dict_model = model.module.state_dict()

        state = {
        'epoch': epoch,
        'state_dict': state_dict_model,
        'optimizer': optimizer.state_dict(),
        }
        
        torch.save(state, outpath+"/pytorch_"+model_name+"_"+str(epoch)+".pt")

        outfile.write("Epoch: {}, Training Accuracy: {:.4f}, Validation Accuracy: {:.4f}\n".format(epoch,train_acc,testing_acc))
        outfile.close()
    return training_loss, training_acc, test_acc

def train_elmo(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,model,test_size,loss_func,X_test,y_test,model_name,cuda_gpu,elmo):
    from allennlp.modules.elmo import batch_to_ids
    import GPUtil
    device_ids = GPUtil.getAvailable(limit = 4)
    device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")
    def pad_tokens(articles_list):
        token_list = []
        for article in articles_list:
            article = article.strip().split()
            if len(article)>sequence_length:
                article = article[:sequence_length]
            token_list.append(article)
        return token_list 

    correct = 0 # set up variables
    total = 0
    step = 0
    training_loss = []
    training_acc = []
    test_acc = []
    for epoch in range(num_epoch):
        for i in range(0,train_size,batch_size):   # gives batch data
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)
            step += 1    # clear gradients for this training step
            
            b_x = elmo(batch_to_ids(pad_tokens(X_train[batch_start:batch_end])))['elmo_representations'][1]
            b_y = torch.from_numpy(y_train[batch_start:batch_end]).long()
            if cuda_gpu:
                b_x = b_x.to(device)
                b_y = b_y.to(device)

            output = model(b_x)               # lstm output
            del b_x
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad() # set the gradiant vector as zero
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()
            if step%100 == 0:
                print("STEP",str(step),"lost is",loss.item()) 
            training_loss.append(loss.item())       
            

        # get train accuracy for each epoch
        for i in range(0,train_size,batch_size):
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)
            b_x = elmo(batch_to_ids(pad_tokens(X_train[batch_start:batch_end])))['elmo_representations'][1]
            if cuda_gpu:
                b_x = b_x.to(device)
                
            train_output = model(b_x)  # lstm output
            del b_x

            if cuda_gpu:
                pred_y = torch.max(train_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(train_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value

            y_true = np.asarray(y_train[batch_start:batch_end]) # reshape training y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the training data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true

        acc = 100.00 * (float(correct) / float(total)) # get accuracy
        print('Training Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        training_acc.append(acc)

        correct = 0
        total = 0
        acc = 0
        for i in range(0,test_size,batch_size): # test every batch
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_test)-1:
                batch_end = len(X_test)
                
            b_x = elmo(batch_to_ids(pad_tokens(X_test[batch_start:batch_end])))['elmo_representations'][1]
            if cuda_gpu:
                b_x = b_x.to(device)

            test_output = model(b_x)  # convert b_x to float
            del b_x
            if cuda_gpu:
                pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            
            y_true = np.asarray(y_test[batch_start:batch_end]) # reshape testing y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the test data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true

        acc = 100.00 * float(correct) / float(total) # get accuracy
        print('Testing Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        test_acc.append(acc)
        #save model afer each epoch
        if torch.cuda.device_count() <= 1:
            state_dict_model = model.state_dict()
        else:
            state_dict_model = model.module.state_dict()
        
        state = {
        'epoch': epoch,
        'state_dict': state_dict_model,
        'optimizer': optimizer.state_dict(),
        }

        torch.save(state, "./output/"+model_name+"/pytorch_"+model_name+"_"+str(epoch)+"")
    return training_loss, training_acc, test_acc

def train_with_bert(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,model,test_size,loss_func,X_test,y_test,model_name,cuda_gpu):
    from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
    import GPUtil
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()

    device_ids = GPUtil.getAvailable(limit = 4)
    device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")

    def bert_embed(batch_sample): 
        embed = torch.zeros([len(batch_sample), sequence_length, 1024], dtype=torch.float32)
        for ind, content in enumerate(batch_sample):
            tokenized_text = bert_tokenizer.tokenize(content)
            tokenized_text = tokenized_text[:min(sequence_length,len(tokenized_text))]
            tokenized_text += ["[PAD]"] * (sequence_length - len(tokenized_text))
            indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = []
            segments_ids += [0] * (sequence_length - len(segments_ids))
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
            embed_last = encoded_layers[-1]
            embed[ind,:,:] = embed_last
        return embed

    correct = 0 # set up variables
    total = 0
    step = 0
    training_loss = []
    training_acc = []
    test_acc = []
    for epoch in range(num_epoch):
        outfile = open("./output/"+model_name+"/pytorch_"+model_name+".txt",'a')
        for i in range(0,train_size,batch_size):   # gives batch data
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)
            print("training")
            step += 1    # clear gradients for this training step
            b_x = bert_embed(X_train[batch_start:batch_end])
            b_y = torch.from_numpy(y_train[batch_start:batch_end])
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)               # lstm output
            del b_x
            loss = loss_func(output, b_y.long())   # cross entropy loss
            optimizer.zero_grad() # set the gradiant vector as zero
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()
            
            if step%1000 == 0:
                print("STEP",str(step),"lost is",loss.item()) 
            training_loss.append(loss.item())       
            

        # get train accuracy for each epoch
        for i in range(0,train_size,batch_size):
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)
            print("training accu")
            b_x = bert_embed(X_train[batch_start:batch_end])
            b_x = b_x.to(device)
            train_output = model(b_x)  # lstm output
            del b_x

            if cuda_gpu:
                pred_y = torch.max(train_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(train_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value

            y_true = np.asarray(y_train[batch_start:batch_end]) # reshape training y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the training data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true

        acc = 100.00 * (float(correct) / float(total)) # get accuracy
        print('Training Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        training_acc.append(acc)
        train_acc = acc

        correct = 0
        total = 0
        acc = 0
        for i in range(0,test_size,batch_size): # test every batch
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_test)-1:
                batch_end = len(X_test)
            print("test")    
            b_x = bert_embed(X_test[batch_start:batch_end])
            b_x = b_x.to(device)
            test_output = model(b_x)  # convert b_x to float
            del b_x
            if cuda_gpu:
                pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            
            y_true = np.asarray(y_test[batch_start:batch_end]) # reshape testing y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the test data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true

        acc = 100.00 * float(correct) / float(total) # get accuracy
        testing_acc = acc
        print('Testing Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        test_acc.append(acc)
        #save model afer each epoch
        if torch.cuda.device_count() <= 1:
            state_dict_model = model.state_dict()
        else:
            state_dict_model = model.module.state_dict()
        
        state = {
        'epoch': epoch,
        'state_dict': state_dict_model,
        'optimizer': optimizer.state_dict(),
        }

        outfile.write("Epoch: {}, Training Accuracy: {:.4f}, Validation Accuracy: {:.4f}\n".format(epoch,train_acc,testing_acc))
        outfile.close()
        torch.save(state, "./output/"+model_name+"/pytorch_"+model_name+"_"+str(epoch)+".pt")
    
    return training_loss, training_acc, test_acc

# define a function for model training with pre-trained word2vec
def train_w2v_final(num_epoch,train_size,batch_size,optimizer,X_train,y_train,sequence_length,embedding_dim,embedding_matrix,model,loss_func,model_name,batch_first,cuda_gpu,outpath):
    if torch.cuda.is_available():
        import GPUtil
        device_ids = GPUtil.getAvailable(limit = 4)
    device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    acc = 0
    step = 0
    training_loss = []
    training_acc = []
    print("training")
    for epoch in range(num_epoch):
        outfile = open(outpath+"/pytorch_"+model_name+".txt",'a')
        # train the model and back-propagate by batch process
        for i in range(0,train_size,batch_size):   # gives batch data
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)

            step += 1
            optimizer.zero_grad()
            if batch_first: 
                input_x = data_prepare.trans2input_batch_first(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            else:
                input_x = data_prepare.trans2input_batch_second(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            
            b_x = torch.from_numpy(input_x).float().to(device)
            b_y = torch.from_numpy(y_train[batch_start:batch_end]).long().to(device)
            del input_x
            output = model(b_x)               # lstm output
            del b_x
            loss = loss_func(output, b_y)   # cross entropy loss
            if step%1000 == 0:
                print("STEP",str(step),"lost is",loss.item())
            
            training_loss.append(loss.item())       
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
                # get train accuracy for each epoch
        for i in range(0,train_size,batch_size):
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)
            if batch_first: 
                input_x = data_prepare.trans2input_batch_first(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            else:
                input_x = data_prepare.trans2input_batch_second(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            
            if cuda_gpu:
                b_x = torch.from_numpy(input_x).float().to(device)  # reshape training x to (batch, time_step, input_size)
            else:
                b_x = torch.from_numpy(input_x).float()   # reshape training x to (batch, time_step, input_size)
            del input_x
            train_output = model(b_x)  # lstm output
            del b_x

            if cuda_gpu:
                pred_y = torch.max(train_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(train_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            
            y_true = np.asarray(y_train[batch_start:batch_end]) # reshape training y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the training data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true
        
        acc = 100.00 * (float(correct) / float(total)) # get accuracy
        print('Training Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        training_acc.append(acc)
        train_acc = acc

        torch.save(model.state_dict(), outpath+"pytorch_"+model_name+"_"+str(epoch)+"") # save the neural network after each epoch
        outfile.write("Epoch: {}, Training Accuracy: {:.4f}".format(epoch,train_acc))
        outfile.close()

    return training_loss, training_acc



def train_w2v_mix(num_epoch,train_size,batch_size,optimizer,X_train,y_train,X_test_auto,y_test_auto,X_test_man,y_test_man,auto_test_size,man_test_size,sequence_length,embedding_dim,embedding_matrix,model,loss_func,model_name,batch_first,cuda_gpu,scheduler,outpath):
    import GPUtil
    device_ids = GPUtil.getAvailable(limit = 4)
    correct = 0
    total = 0
    acc = 0
    step = 0
    training_loss = []
    auto_acc = []
    man_acc = []
    print("training")
    for epoch in range(num_epoch):
        outfile = open(outpath+"pytorch_"+model_name+".txt",'a')
        scheduler.step()
        # train the model and back-propagate by batch process
        for i in range(0,train_size,batch_size):   # gives batch data
            batch_start = i
            batch_end = i+batch_size
            if batch_end > len(X_train)-1:
                batch_end = len(X_train)

            step += 1
            optimizer.zero_grad()
            if batch_first: 
                input_x = data_prepare.trans2input_batch_first(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            else:
                input_x = data_prepare.trans2input_batch_second(X_train[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            
            if cuda_gpu:
                b_x = torch.from_numpy(input_x).float().cuda(device_ids[0])   # reshape training x to (batch, time_step, input_size)
                b_y = torch.from_numpy(y_train[batch_start:batch_end]).long().cuda(device_ids[0]) # reshape training y to (batch, time_step, input_size)
            else:
                b_x = torch.from_numpy(input_x).float()   # reshape training x to (batch, time_step, input_size)
                b_y = torch.from_numpy(y_train[batch_start:batch_end]).long() # reshape training y to (batch, time_step, input_size)
            del input_x
            output = model(b_x)               # lstm output
            del b_x
            loss = loss_func(output, b_y)   # cross entropy loss
            if step%100 == 0:
                print("STEP",str(step),"lost is",loss.item())
            
            training_loss.append(loss.item())       
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
        # get auto test accuracy for each epoch
        for i in range(0,auto_test_size,batch_size):
            batch_start = i
            batch_end = i+batch_size
            if batch_end > auto_test_size-1:
                batch_end = auto_test_size
            if batch_first: 
                input_x = data_prepare.trans2input_batch_first(X_test_auto[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            else:
                input_x = data_prepare.trans2input_batch_second(X_test_auto[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            
            if cuda_gpu:
                b_x = torch.from_numpy(input_x).float().cuda(device_ids[0])   # reshape training x to (batch, time_step, input_size)
            else:
                b_x = torch.from_numpy(input_x).float()   # reshape training x to (batch, time_step, input_size)
            del input_x
            test_output = model(b_x)  # lstm output
            del b_x

            if cuda_gpu:
                pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            
            y_true = np.asarray(y_test_auto[batch_start:batch_end]) # reshape training y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the training data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true
        
        acc = 100.00 * (float(correct) / float(total)) # get accuracy
        print('Auto label Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        auto_acc.append(acc)
        auto_test_acc = acc
        # get manual test accuracy for each epoch
        correct = 0
        total = 0
        acc = 0
        for i in range(0,man_test_size,batch_size):
            batch_start = i
            batch_end = i+batch_size
            if batch_end > man_test_size-1:
                batch_end = man_test_size

            if batch_first: 
                input_x = data_prepare.trans2input_batch_first(X_test_man[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            else:
                input_x = data_prepare.trans2input_batch_second(X_test_man[batch_start:batch_end,:],batch_size,sequence_length,embedding_dim,embedding_matrix) # word embedding using google news vectors
            
            if cuda_gpu:
                b_x = torch.from_numpy(input_x).float().cuda(device_ids[0])   # reshape training x to (batch, time_step, input_size)
            else:
                b_x = torch.from_numpy(input_x).float()   # reshape training x to (batch, time_step, input_size)
            del input_x
            test_output = model(b_x)  # lstm output
            del b_x

            if cuda_gpu:
                pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze() 
            
            y_true = np.asarray(y_test_man[batch_start:batch_end]) # reshape testing y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the test data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true
        
        acc = 100.00 * (float(correct) / float(total))# get accuracy
        print('Manual Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        man_acc.append(acc)
        man_test_acc = acc
        # save model afer each epoch
        if torch.cuda.device_count() <= 1:
            state_dict_model = model.state_dict()
        else:
            state_dict_model = model.module.state_dict()
        
        state = {
        'epoch': epoch,
        'state_dict': state_dict_model,
        'optimizer': optimizer.state_dict(),
        }
        torch.save(state, outpath+"/pytorch_"+model_name+"_"+str(epoch)+".pt")
        outfile.write("Epoch: {}, Auto Accuracy: {:.4f}, Manual Accuracy: {:.4f}\n".format(epoch,auto_test_acc,man_test_acc))
        outfile.close()
    return training_loss, auto_acc, man_acc


def train_bert_mix(num_epoch,train_size,batch_size,optimizer,X_train,y_train,X_test_auto,y_test_auto,X_test_man,y_test_man,auto_test_size,man_test_size,sequence_length,embedding_dim,model,loss_func,model_name,batch_first,cuda_gpu,outpath):
    from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
    import GPUtil
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bert_model = BertModel.from_pretrained('bert-large-uncased')
    bert_model.eval()

    device_ids = GPUtil.getAvailable(limit = 4)
    device = torch.device("cuda:"+str(device_ids[0])+"" if torch.cuda.is_available() else "cpu")

    def bert_embed(batch_sample): 
        embed = torch.zeros([len(batch_sample), sequence_length, embedding_dim], dtype=torch.float32)
        for ind, content in enumerate(batch_sample):
            tokenized_text = bert_tokenizer.tokenize(content)
            tokenized_text = tokenized_text[:min(sequence_length,len(tokenized_text))]
            tokenized_text += ["[PAD]"] * (sequence_length - len(tokenized_text))
            indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = []
            segments_ids += [0] * (sequence_length - len(segments_ids))
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
            embed_last = encoded_layers[-1]
            embed[ind,:,:] = embed_last
        return embed

    correct = 0 # set up variables
    total = 0
    step = 0
    training_loss = []
    auto_acc = []
    man_acc = []
    for epoch in range(num_epoch):
        start = time()
        outfile = open(outpath+"/pytorch_"+model_name+".txt",'a')
        for i in range(0,train_size,batch_size):   # gives batch data
            batch_start = i
            batch_end = i+batch_size
            if batch_end > train_size-1:
                batch_end = train_size
            step += 1    # clear gradients for this training step
            b_x = bert_embed(X_train[batch_start:batch_end])
            b_y = torch.from_numpy(y_train[batch_start:batch_end])
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)               # lstm output
            del b_x
            loss = loss_func(output, b_y.long())   # cross entropy loss
            optimizer.zero_grad() # set the gradiant vector as zero
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()
            
            if step%1000 == 0:
                print("STEP",str(step),"lost is",loss.item()) 
            training_loss.append(loss.item())       
            
        # get train accuracy for each epoch
        for i in range(0,auto_test_size,batch_size):
            batch_start = i
            batch_end = i+batch_size
            if batch_end > auto_test_size-1:
                batch_end = auto_test_size
            b_x = bert_embed(X_test_auto[batch_start:batch_end])
            b_x = b_x.to(device)
            train_output = model(b_x)  # lstm output
            del b_x

            if cuda_gpu:
                pred_y = torch.max(train_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(train_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value

            y_true = np.asarray(y_test_auto[batch_start:batch_end]) # reshape training y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the training data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true

        acc = 100.00 * (float(correct) / float(total)) # get accuracy
        print('Auto Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        auto_acc.append(acc)
        auto_test_acc = acc

        correct = 0
        total = 0
        acc = 0
        for i in range(0,man_test_size,batch_size): # test every batch
            batch_start = i
            batch_end = i+batch_size
            if batch_end > man_test_size-1:
                batch_end = man_test_size   
            b_x = bert_embed(X_test_man[batch_start:batch_end])
            b_x = b_x.to(device)
            test_output = model(b_x)  # convert b_x to float
            del b_x
            if cuda_gpu:
                pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze() # to get the maximum probability of very article in the batch and get its value
            
            y_true = np.asarray(y_test_man[batch_start:batch_end]) # reshape testing y to (batch, time_step, input_size)
            total += y_true.shape[0] # the total number of article in the test data
            correct += (pred_y == y_true).sum().item() # the number of intances that pred_y matches with the y_true

        acc = 100.00 * float(correct) / float(total) # get accuracy
        print('Manual Accuracy of the network on epoch %i: %d %%' % (epoch, acc))
        man_acc.append(acc)
        man_test_acc = acc
        #save model afer each epoch
        if torch.cuda.device_count() <= 1:
            state_dict_model = model.state_dict()
        else:
            state_dict_model = model.module.state_dict()
        
        state = {
        'epoch': epoch,
        'state_dict': state_dict_model,
        'optimizer': optimizer.state_dict(),
        }
        end = time()
        used_time = end - start
        outfile.write("Epoch: {}, Run time: {:.4f}, Auto Accuracy: {:.4f}, Manual Accuracy: {:.4f}\n".format(epoch,used_time, auto_test_acc,man_test_acc))
        outfile.close()
        torch.save(state, outpath+"/pytorch_"+model_name+"_"+str(epoch)+".pt")
    
    return training_loss, auto_acc, man_acc

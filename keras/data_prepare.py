# the following codes aree packaged as a py file that can be import in other python scripts. 
# this user own package is named as "data_prepare"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
# function for text cleaning
def clean_text(text):
    ## Remove puncuation
    text = text.translate(string.punctuation)
    ## Convert words to lower case and split them
    # replace non-readable apostrophes
    # replace contractions of sequences as its original form .
    text = text.lower().replace("′", "'").replace("’", "'")\
   .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
   .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
   .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
   .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
   .replace("'ll", " will")
    tokens = word_tokenize(text)    
    words = [word for word in tokens if word.isalpha()] # remove non-alphabatical words

    ## Remove stop words
    stops = set(stopwords.words("english"))
    word_filter = [w for w in words if not w in stops and len(w) >= 3]
    text_return = " ".join(word_filter)
    return text_return
# function for data loading
def load_data(dataset,data_size):
    articles = []
    i = 0
    with open(dataset) as f:
        lines = f.readlines()
        for item in lines[:data_size]:
            #print(item)
            seq = clean_text(item)
            articles.append(seq)
            i += 1
            if i%10000 == 0:
                print(str(i))
    return articles
# function for labels loading
def load_labels(labelset,data_size):
    labels = []
    with open(labelset) as f:
        lines = f.readlines()
        for item in lines[:data_size]:
            labels.append(int(item.split("\n")[0]))
    labels = np.asarray(labels)
    return labels
# function for tokenization and padding
def tokenize(articles,vocabulary_size,sequence_length,load_model,tokenizer=''):
    if load_model == False:
        tokenizer = Tokenizer(num_words= vocabulary_size)
        tokenizer.fit_on_texts(articles) # create a dictionary that has words in articles as key and the index as value
    sequences = tokenizer.texts_to_sequences(articles)
    data = pad_sequences(sequences, maxlen = sequence_length) # pad the article that has less words than the sequence_length to sequence_length using zeros
    return data,tokenizer
# function for data spliting
def split_data(data,label_array,train_size):
    X_train = data[0:train_size,:]
    y_train = label_array[0:train_size]

    X_test = data[train_size:,:]
    y_test = label_array[train_size:]
    return X_train, y_train, X_test, y_test
# call data prepare functions and return results
def data_ready(dataset, labelset,data_size,vocabulary_size,sequence_length,train_size,load_model,tokenizer=''):
    articles = load_data(dataset,data_size)
    label_array = load_labels(labelset,data_size)
    data,tokenizer = tokenize(articles,vocabulary_size,sequence_length,load_model,tokenizer='')
    X_train, y_train, X_test, y_test = split_data(data, label_array, train_size)
    return X_train, y_train, X_test, y_test, tokenizer

# function for pre-trained word embedding loading
# tokenizer is needed here to map the word in google w2v file with the words in the articles.
def load_w2v(w2v_file, binary, vocabulary_size, embedding_dim,tokenizer):
    word_vectors = KeyedVectors.load_word2vec_format(w2v_file, binary=binary)
    w2v = word_vectors

    embeddings_index = dict()
    vocab = w2v.vocab.keys()
    for word in vocab:
        coefs = np.asarray(w2v.word_vec(word), dtype='float32')
        embeddings_index[word] = coefs

    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            continue
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix
# function for word transform
# to get the vector of all words in articles of batch_size
def trans2input(batch_data, batch_size, sequence_length,embedding_dim, embedding_matrix):
    zero_vc = np.zeros(embedding_dim,dtype='float32')
    input_data = np.zeros((batch_size, sequence_length,embedding_dim))
    for i in range(batch_size):
        # print(i)
        for j in range(sequence_length):
            indx = batch_data[i,j]
            if indx != 0:
                input_data[i,j,:] = embedding_matrix[indx]
            else:
                input_data[i,j,:] = zero_vc
    return input_data

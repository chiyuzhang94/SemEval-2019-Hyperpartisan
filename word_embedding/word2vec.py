## Word2Vec using Gensim for HW03 Crated on Nov6###
import nltk
from nltk.corpus import stopwords
WPT = nltk.WordPunctTokenizer()
import string
import multiprocessing
from gensim.models import word2vec
import itertools
## set the parameters
epoch = 3
feature_size = 300 # Word vector dimensionality
window_context = 10 # Context window size
min_word_count = 50 # Minimum word count
batch_size = 10000
data_size = 1000000
### Normalizing Tokens ##
def norm_doc_tokenizer (input_file,n,batch_size):
    i = 0
    with open (input_file, 'r') as text_file:
        for doc in itertools.islice(text_file, n, n+batch_size):
            i += 1
            ## lower case and remove special chars and white spaces
            #dcc = re.sub(r'[^a-zA-Z0-9\s]', '', doc)
            doc = doc.lower().replace("′", "'").replace("’", "'")\
            .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
            .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
            .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
            .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
            .replace("'ll", " will")
            doc = doc.strip()
            ## tokenize documents
            tokens = WPT.tokenize(doc)
            ## remove the punctuation from the tokenized words
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            ## only keep the English words
            words = [wd for wd in stripped if wd.isalpha()]
            if i%10000 == 0:
                print(str(i) + " samples")
            yield words
for ep in range(epoch):
    step = 0
    for i in range(0,data_size,batch_size):
        step += 1
        if i == 0 and ep == 0:
        ## run the gensim word2vev ###
            tokenized_corpus = list(norm_doc_tokenizer("../data/full_dataset.txt",i,batch_size))
            w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size, window=window_context, min_count = min_word_count,workers=multiprocessing.cpu_count())
        else:
            tokenized_corpus = list(norm_doc_tokenizer("../data/full_dataset.txt",i,batch_size))
            w2v_model.train(tokenized_corpus, total_examples=batch_size, epochs=w2v_model.epochs)
        print("Epoch",str(ep+1),",Step",str(step))

#w2v_model.save("./output/word2vec_gensim")
w2v_model.wv.save_word2vec_format("./output/word2vec_org","./output/w2v_vocabulary", binary=False)
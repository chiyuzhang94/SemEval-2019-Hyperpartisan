## Created by Xuejun Ryan Ji                                                    ##
## Created on Oct11 and 12 2018                                                   ##
## Updated on Oct20                                                              ##
## Purspose: create the word and sentence frequency dist and summary statistics ##

## Data preprocess
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import string
def clean_text(text):
    ## Remove puncuation
    
    ## Convert words to lower case and split them
    # replace non-readable apostrophes
    # replace contractions of sequences as its original form .
    text = text.lower().replace("′", "'").replace("’", "'")\
    .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
    .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
    .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
    .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
    .replace("'ll", " will")
    text = text.translate(string.punctuation)
    tokens = word_tokenize(text)    
    words = [word for word in tokens if word.isalpha()] # remove non-alphabatical words

    ## Remove stop words
    stops = set(stopwords.words("english"))
    word_filter = [w for w in words if not w in stops and len(w) >= 3]
    return word_filter

## define word prep for word count
def word_prep(input_file):
    i=0
    with open (input_file, 'r') as f:
        for i, line in enumerate (f):
            areticle = clean_text(line) 
            if i % 1000 == 0:
              print(str(i+1)," samples")
            yield areticle

data_file='training_set.txt'
documents_word = list (word_prep (data_file))

""" Sparsity = the percentage of documents that this terms occurs to the entire corpus 
    len(doc_term)/len(corpus)"""

sparsity_dict = {}
for docterm in documents_word:
  for word in docterm:
    if word in sparsity_dict:
      continue
    else:
      sparsity_dict[word] = 0

num_total=len(documents_word)

for word in sparsity_dict:
  for docterm in documents_word:
    if word in docterm:
      sparsity_dict[word] += 1
    else:
      continue

for word in sparsity_dict:
  sparsity_dict[word] = float(sparsity_dict[word])/float(num_total)


with open('word_sparsity.json', 'w') as f:
    json.dump(sparsity_dict, f)
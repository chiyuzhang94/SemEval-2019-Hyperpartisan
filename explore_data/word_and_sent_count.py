'''
input: all aritcles
count the length of words and sentences of articels 
'''
import os
for root, dirs, files in os.walk("."):
    for filename in files:
        print(filename)

## Data preprocess
import nltk
import gensim
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import string
import logging
from matplotlib import pyplot as plt
import seaborn as sns
import json
#please enter you dataset name 
data_file='training_dub.txt'
####
stopwords = stopwords.words('english')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

## Data Preprocess
## define word prep for word count
def token_prep(input_file):
  article_word = []
  articel_sent = []
  logging.info("reading file {0}...".format(input_file))
  with open (input_file, 'r') as f:
    for i, line in enumerate (f):
      tokens=gensim.utils.simple_preprocess (line, deacc=False, min_len=2, max_len=15)
      filtered = [word for word in tokens if word not in stopwords]  # remove stopwords
      sents=sent_tokenize(line)
      article_word.append(filtered)
      articel_sent.append(sents)

      if (i%10000==0):
        logging.info ("read {0} articles".format (i))
  return article_word, articel_sent


documents_word, documents_sent = token_prep(data_file)


# calculate the number of words in each article
words_num_list=[]
sents_num_list=[]

i = 0
while i < len(documents_word):
     num=len(documents_word[i])
     words_num_list.append(num)
     i+=1

# calculate the number of sentences in each article
idx = 0
while idx < len(documents_sent):
     num=len(documents_sent[idx])
     sents_num_list.append(num)
     idx+=1

#change df to panda
df1 = pd.DataFrame({'word_freq':words_num_list})
df2 = pd.DataFrame({'sent_freq':sents_num_list})

#Write the results to the file
with open ('summary_results.txt', 'w') as f:
         print("Word Counts Info:",
               "total:",df1['word_freq'].sum(),  ### total words
               "Summary:", df1['word_freq'].describe(), ### averge counts for each article
               "XLarge:", df1.sort_values(['word_freq'],ascending=False).head(20).reset_index(drop=True),
               "Sentence Counts Info:",
               "total:",df2['sent_freq'].sum(),  ### total num
               "Summary:", df2['sent_freq'].describe(), ### averge counts for each article
               "XLarge:", df2.sort_values(['sent_freq'],ascending=False).head(20).reset_index(drop=True),
                ### the counts of words extra long article
               file=f, sep="\n")

#store data into json files
with open('word_count.json', 'w') as outfile1:
    json.dump(words_num_list, outfile1)

with open('sentence_count.json', 'w') as outfile2:
    json.dump(sents_num_list, outfile2)
###plot the result

plt.figure(figsize=(8,15))

plt.subplot(4,1,1)
sns.violinplot(df1['word_freq'])
plt.xlabel(" ")
plt.title("Distribution of Word and Sentence Counts")


plt.subplot(4,1,2)
sns.distplot(df1['word_freq'])
plt.xlabel("word frequency")

plt.subplot(4,1,3)
sns.violinplot(df2['sent_freq'])
plt.xlabel(" ")

plt.subplot(4,1,4)
sns.distplot(df2['sent_freq'])
plt.xlabel("sentence frequency")
plt.savefig("word_and sentence dist.png")   ## save the results

plt.close()



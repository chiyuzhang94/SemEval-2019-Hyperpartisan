## Created by Xuejun Ryan Ji                                                    ##
## Created on Oct11 and 12 2018                                                   ##
## Updated on Oct12                                                              ##
## Purspose: create the word and sentence frequency dist and summary statistics ##

## Data preprocess
import nltk
import gensim
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from nltk.tokenize import sent_tokenize
import itertools
from collections import Counter
import seaborn as sns
import pandas as pd
import json

## Data Preprocess
## define word prep for word count
def word_prep(input_file):

    with open (input_file, 'r') as f:
        for i, line in enumerate(f):
            tokens=gensim.utils.simple_preprocess (line, deacc=False, min_len=2, max_len=15)
            filtered = [word for word in tokens if word not in stopwords]  # remove stopwords
            if i % 10000 == 0: # remove this syntax when it is run in cluster
                print("sampel: ", str(i))

            yield filtered

data_file='training_set.txt'
documents_word = list (word_prep (data_file))
#len(documents_word)

### flatten list of list into list

flattened_list  = list(itertools.chain(*documents_word))

#len(flattened_list)

### create a word count dict
cnt_list=Counter(flattened_list).most_common()     
cnt_dict = Counter(flattened_list)             #Extract a word frequency dictionary of the data
freq_num = []
for item in cnt_list:
    freq_num.append(item[1])

wfd_dic = dict()
# count the distribution of frequency number
freq_count = Counter(freq_num).most_common()

for count_num in freq_count:
    wfd_dic[count_num[0]] = count_num[1]

with open('word_freq_dis.json', 'w') as outfile1:
    json.dump(wfd_dic, outfile1)


df = pd.DataFrame.from_dict(cnt_dict, orient='index').reset_index()
df = df.rename(columns={'index':'words', 0:'count'})
df.head()


#### Print out the statisitcs
with open ('unique_word_frequency.txt', 'w') as f:
         print("The number of unique words:",
               len(cnt_dict),  ### total words
               "Descriptive Statistics of Word Freq",
               "Total：", df['count'].sum(),
               "Summary:", df['count'].describe(), ### averge counts for each article
               "XLarge:", df.sort_values(['count'],ascending=False).head(20).reset_index(drop=True),
                ### the counts of words extra long article
               file=f, sep="\n")

##### Plot the dist of the word freq
from matplotlib import pyplot as plt
sns.distplot(df['count'])
plt.xlabel("unique word frequency")
plt.savefig("uninque_word_freq.png")   ## save the results
plt.close()

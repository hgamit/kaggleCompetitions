'''
Created on Apr 4, 2017
@author: HGamit
'''

import matplotlib.cm as cm
import math
import pandas as pd
import numpy as np
import re  #regular expressions
#import gensim
import matplotlib.pylab as plt
from nltk.corpus import stopwords, wordnet
import os
from nltk.tokenize import sent_tokenize
from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

eng_stopwords = set(stopwords.words('english'))  

import time
start_time = time.time()

replacements = {'one':'1', 
                'two':'2', 
                'three':'3', 
                'four':'4', 
                'five':'5', 
                'six':'6', 
                'seven':'7', 
                'eight':'8', 
                'nine':'9', 
                'ten':'10'}


os.chdir("D:\\Build\ML\\comp\\indeed")

#os.chdir("D:\\Machine\\hacker-indeed")

train_df = pd.read_table("train.tsv", nrows=5)
#print(train_df.head())
#train_df = train_df.head()

class Test(object):
    '''
    Natural language text processing definitions
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def corpus_synonym(self, search_words):
        synonyms = []
        for i in search_words:
            for syn in wordnet.synsets(i):
                for l in syn.lemmas():
                    synonyms.append(l.name())
        return (set(synonyms))
    
    def matched_sentences(self, search_words, text):
        sntences = []
        sent_tokenize_list = sent_tokenize(text)
        for i in search_words:
            for j in sent_tokenize_list:
                if(i in j):
                    sntences.append(j)
        return (set(sntences))
        
    def iterate_data(self,df):
        text = ""
        for row in df.iterrows():
            text += row[1]['description']
        
        return text
    
    def replace(self, match):
        return replacements[match.group(0)]
    

    def RepresentsInt(self, s):
        try: 
            int(s)
            return True
        except ValueError:
            return False


    def remove_numbers(self, string):
        words = string.split()
        for word in words:
            if(self.RepresentsInt(word) and int(word)>50):
                string = re.sub(word, ' ', string)
        string = re.sub(' +', ' ', string)
        return string


    def basic_cleaning(self, string):
        string = str(string)
        string = string.lower()
        #print(string.encode("ascii", "ignore"))
        string = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replacements), self.replace, string)        
        string = re.sub(r'\byear\b', ' years ', string)
        string = re.sub(r'\b([0-9]).?plus\b', r' \1+ ', string)
        string = re.sub(r'\bmonth\b', 'months', string)
        string = re.sub(r'\b([0-9]).?to.?([0-9])\b', r' \1-\2 ', string)
        string = re.sub(r'\b([0-9]) +\- +([0-9])\b', r' \1-\2 ', string)
        string = re.sub(r'\b(bachelor(s|.s)?|baccalaureate|(bs|ba)|college|undergraduate|engineering|accounting)( degree)?\b', ' bachelors degree ', string)
        string = re.sub(r'\b(master(s|.s)?|(ms|ma|mba))( degree)?\b', 'masters degree', string)
        string = re.sub(r'\b(md|phd|pharmd)\b', 'phd', string)
        string = re.sub(r'\b(full time)\b', 'full-time', string)
        string = re.sub(r'\b(part time)\b', 'part-time', string)
        string = re.sub(r'\b(associate(s|.s)? |high school |general education |(adn|ged|diploma))(diploma|degree)?\b', ' associates degree ', string)
        string = re.sub(r'\b(licence|license)\b', ' license ', string)
        string = re.sub('[\(\)\_\^\%\$\.\,\\\!\&\/\*\:\;\'\'\"\{\}\[\]\\/]+', ' ', string)
        searchStr = "((at least|minimum)? ?(([0-9])(\+?|\-[0-9])?) (months|years) ?\+?)"
        match = re.search(searchStr, string)
        #print(match)
        num = ''
        gr6=''
        if match:
            wh = match.group(1)
            num = match.group(4)
            gr6 = match.group(6)
        if(num != '' and gr6!= ''):
            if(int(num)>4 and str(gr6)=='years'):
                string = re.sub(wh, r' 5+ years ', string)
            elif(int(num)>=2 and str(gr6) =='years'):
                string = re.sub(wh,' 2-4 years ', string)
            else:
                string = re.sub(wh, ' 1+ years ', string)
        
        string = ' '.join([i for i in string.split() if i not in ["a", "and", "of", "the", "to", "on", "in", "at", "is"] and i not in eng_stopwords])    
        string = self.remove_numbers(string)
        string = ' '.join([lem.lemmatize(i, "v") for i in string.split()])
        string = re.sub(' +', ' ', string)
        #print(string.encode("ascii", "ignore"))
        return string
    
    def _removeNonAscii(self, s):
        return "".join(i for i in s if ord(i)<128)
    
myCall = Test()

s_list = ['licence']
#search_words = myCall.corpus_synonym(s_list)
#print(search_words)
search_words = ['licence', 'license']


train_df['part-time-job']=0
train_df['licence-needed']=0
train_df['full-time-job']=0
train_df['hourly-wage']=0
train_df['salary']=0
train_df['associate-needed']=0
train_df['bs-degree-needed']=0
train_df['ms-or-phd-needed']=0
train_df['1-year-experience-needed']=0
train_df['2-4-years-experience-needed']=0
train_df['5-plus-years-experience-needed']=0
train_df['supervising-job']=0


for i,row in train_df.iterrows():
    words = str(row['tags']).split()        
    for w in words:
        if(w == 'part-time-job'):
            train_df.set_value(i,'part-time-job','1')
        if(w == 'full-time-job'):
            train_df.set_value(i,'full-time-job','1')
        if(w == 'licence-needed'):
            train_df.set_value(i,'licence-needed','1')
        if(w == 'supervising-job'):
            train_df.set_value(i,'supervising-job','1')
        if(w == '5-plus-years-experience-needed'):
            train_df.set_value(i,'5-plus-years-experience-needed','1')
        if(w == '2-4-years-experience-needed'):
            train_df.set_value(i,'2-4-years-experience-needed','1')
        if(w == '1-year-experience-needed'):
            train_df.set_value(i,'1-year-experience-needed','1')
        if(w == 'ms-or-phd-needed'):
            train_df.set_value(i,'ms-or-phd-needed','1')
        if(w == 'bs-degree-needed'):
            train_df.set_value(i,'bs-degree-needed','1')
        if(w == 'associate-needed'):
            train_df.set_value(i,'associate-needed','1')
        if(w == 'salary'):
            train_df.set_value(i,'salary','1')
        if(w == 'hourly-wage'):
            train_df.set_value(i,'hourly-wage','1')



x_train = pd.DataFrame(data = {'text':[], 'label':[]})

d = {}


#with open("file.txt", "a") as text_file:
for i,row in train_df.iterrows():
        #sent_tokenize_list = sent_tokenize(myCall.basic_cleaning(row[1]['description']))
        #print(row[1]['description'].encode("ascii", "ignore"))
        #print("{}".format(row[1]['description'].encode("ascii", "ignore")), file=text_file)
    #print (row['description'].encode("ascii", "ignore"))
    tra = myCall.basic_cleaning(row['description'])
    #val = [row[1]['licence-needed'], tra.encode("ascii", "ignore")]
    d[tra.encode("ascii", "ignore")] = row['licence-needed']
    #x_train.loc[len(x_train)]= val
        #x_train.append({'text':tra.encode("ascii", "ignore"), 'label': row[1]['licence-needed']}, ignore_index=True)
        #print("{}".format(tra.encode("ascii", "ignore")), file=text_file)

    
    #nf = [x for x in search_words if x in desc.split()]
    #if(len(nf)):
                #train_dict[se.encode("ascii", "ignore")] = 1
        #matched = ' '.join([i for i in nf])
        #val = [matched, '1']
        #print ("Match:" )
        #print (desc.encode("ascii", "ignore"))
    #else:
                #train_dict[se.encode("ascii", "ignore")] = 0
        #val = ["no-tag", '0']
        #print ("No Match:")
        #print (desc.encode("ascii", "ignore"))
        
   # x_train.loc[len(x_train)]= val
                
    #for i in myCall.matched_sentences(search_words,myCall.basic_cleaning(row[1]['description'])):
        #print(i.encode("ascii", "ignore"))
        #val = [i.encode("ascii", "ignore"), '1']
        #x_train.loc[len(x_train)]= val
        #x_train.append({'sent_match':'test1', 'licence-needed': 'test2'}, ignore_index=True)

def word_feats(key):
    return re.sub(r'^b\'|\'', ' ', str(key))




#print("DIC:")
#print(d)

newd = dict()
for key, value in d.items():
    newd[str(key)] = value


import itertools
n = int(len(d)*0.9) // 2          # length of smaller half
i = iter(d.items()) 
d1 = dict(itertools.islice(i, n))   # grab first n items
d2 = dict(i)                        # grab the rest

#train_set = dict(d.items()[len(d)*9/10:])
#test_set = dict(d.items[:len(d)/10])

newd1 = [(word_feats(f), d1[f]) for f in d1 ]
newd2 = [(word_feats(f), d2[f]) for f in d2 ]


from nltk.tokenize import word_tokenize # or use some other tokenizer
all_words1 = set(word.lower() for passage in newd1 for word in word_tokenize(passage[0]))
train_hashable = [({word: (word in word_tokenize(x[0])) for word in all_words1}, x[1]) for x in newd1]

all_words2 = set(word.lower() for passage in newd2 for word in word_tokenize(passage[0]))
test_hashable = [({word: (word in word_tokenize(x[0])) for word in all_words2}, x[1]) for x in newd2]

print(train_hashable)
print(test_hashable)

#train_set, test_set = x_train[:3937], x_train[437:]
import nltk
classifier = nltk.NaiveBayesClassifier.train(train_hashable)


print("most informative features:")
print(classifier.show_most_informative_features())

print("Accuracy:")

print(nltk.classify.accuracy(classifier, test_hashable))

#out_train = train_set.to_json(orient='records')


#all_words = set(word.lower() for passage in newd1 for word in word_tokenize(passage[0]))
#test_hashable = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in newd1]


#with open('license_train.json', 'w') as f:
 #   f.write(out_train)
    
#out_test = test_set.to_json(orient='records')

#with open('license_test.json', 'w') as f:
  #  f.write(out_test)

#with open('license.json', 'r') as fp:
 #   cl = NBC(fp, format="json")

print("Time Taken: --- %s seconds ---" % (time.time() - start_time))
#model = NBC(train_dict)

#print(x_train)
#for i in myCall.matched_sentences(search_words,myCall.basic_cleaning(myCall.iterate_data(train_df))):
    #myCall._removeNonAscii(i.encode("utf-8"))
    #print(i.encode("ascii", "ignore"))
    #with open("D:\\Output.txt", "a") as text_file:
        #print("{}".format(i.encode("ascii", "ignore")), file=text_file)
#print(myCall.corpus_synonym(search_words))

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
eng_stopwords = set(stopwords.words('english'))  

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


os.chdir("D:\\Build\\ML\\comp\\indeed")

train_df = pd.read_table("train.tsv")
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
    
    def basic_cleaning(self, string):
        string = str(string)
        string = string.lower()
        string = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replacements), self.replace, string) 
        string = re.sub('[\(\)\_\^\%\$\.\\+\/\*\'\'\"\{\}\[\]\\/]+', ' ', string)
        string = ' '.join([i for i in string.split() if i not in ["a", "and", "of", "the", "to", "on", "in", "at", "is"] and i not in eng_stopwords])
        string = re.sub(' +', ' ', string)
        return string
    
    def _removeNonAscii(self, s):
        return "".join(i for i in s if ord(i)<128)
    
    def create_pred_variables(self,train_df):
        
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
                if(w == 'full-time-job'):
                    train_df.set_value(i,'full-time-job','1')
                    
    def feature_extractor(self):
        new =""
                    
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
train_df['licence-word']=0


for i,row in train_df.iterrows():
    words = str(row['tags']).split()
    desc = myCall.basic_cleaning(str(row['description']))
    nf = [x for x in search_words if x in desc.split()]
    if(len(nf)>0):
        train_df.set_value(i,'licence-word','1')
        
    for w in words:
        if(w == 'part-time-job'):
            train_df.set_value(i,'part-time-job','1')
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
        if(w == 'full-time-job'):
            train_df.set_value(i,'full-time-job','1')


x_train = pd.DataFrame(data = {'text':[], 'label':[]})

#train_dict = {}

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def remove_numbers(string):
    words = string.split()
    for word in words:
        if(RepresentsInt(word) and int(word)>50):
            string = re.sub(word, ' ', string)
    string = re.sub(' +', ' ', string)
    return string



for row in train_df.iterrows():
    #sent_tokenize_list = sent_tokenize(myCall.basic_cleaning(row[1]['description']))
    tra = myCall.basic_cleaning(row[1]['description'])
    with open("file.txt", "w") as text_file:
        print("{}".format(tra.encode("ascii", "ignore")), file=text_file)
    
    
    
    
    
    
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

#print(x_train)

out = x_train.to_json(orient='records')

with open('license.json', 'w') as f:
    f.write(out)

with open('license.json', 'r') as fp:
    cl = NBC(fp, format="json")

#import json
#with open('result.json', 'w') as fp:
#    json.dump(train_dict, fp)

#model = NBC(train_dict)

#print(x_train)
#for i in myCall.matched_sentences(search_words,myCall.basic_cleaning(myCall.iterate_data(train_df))):
    #myCall._removeNonAscii(i.encode("utf-8"))
    #print(i.encode("ascii", "ignore"))
    #with open("D:\\Output.txt", "a") as text_file:
        #print("{}".format(i.encode("ascii", "ignore")), file=text_file)
#print(myCall.corpus_synonym(search_words))


def basic_cleaning(string):
    string = str(string)
    string = string.lower()
    string = re.sub('year', ' years ', string)
	string = re.sub('month', ' months ', string)
	string = re.sub('([0-9]*).?to.?([0-9]*)', ' \2-\3 ', string)
    string = re.sub('([0-9]*) *\- *([0-9]*)', ' \2-\3 ', string)
    match = re.search('((at least|Minimum)? ?([0-9]*(\+?|\-[0-9]*)?) (months|years) ?\+?)', string)
    num = match.group(2) if match else None
	gr5 = match.group(5) if match else None
    if(num>4 and gr5=='years'):
        string = re.sub('((at least|Minimum)? ?([0-9]*(\+?|\-[0-9]*)?) (months|years) ?\+?)', ' 5+ years ', string)
    elif(num>2 and gr5=='years'):
        string = re.sub('((at least|Minimum)? ?([0-9]*(\+?|\-[0-9]*)?) (months|years) ?\+?)', ' 2-4 years ', string)
    else:
        string = re.sub('((at least|Minimum)? ?([0-9]*(\+?|\-[0-9]*)?) (months|years) ?\+?)', ' 1+ years ', string)
    string = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replacements), replace, string) 
    string = re.sub('(bachelor(s|.s)?|baccalaureate|(bs|ba)|college|undergraduate|engineering|accounting)( degree)?', ' bachelors degree ', string)
    string = re.sub('(master(s|.s)?|(ms|ma|mba))( degree)?', ' masters degree ', string)
    string = re.sub('(md|phd|pharmd)', ' phd ', string)
    string = re.sub('(associate(s|.s)? |high school |general education |(adn|ged|diploma))(diploma|degree)?', ' associates degree ', string)
    string = re.sub('(licence|license)', ' license ', string)
    string = re.sub('[\(\)\_\^\%\$\.\,\\\!\&\/\*\:\;\'\'\"\{\}\[\]\\/]+', ' ', string)
    string = ' '.join([i for i in string.split() if i not in ["a", "and", "of", "the", "to", "on", "in", "at", "is"] and i not in eng_stopwords])
    string = remove_numbers(string)
    string = re.sub(' +', ' ', string)
    return string

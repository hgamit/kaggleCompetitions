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
        string = re.sub('[\(\)\_\^\%\$\*\'\'\"\{\}\[\]\\/]+', ' ', string)
        string = ' '.join([i for i in string.split() if i not in ["a", "and", "of", "the", "to", "on", "in", "at", "is"]])
        string = re.sub(' +', ' ', string)
        return string
    
    def _removeNonAscii(self, s):
        return "".join(i for i in s if ord(i)<128)

myCall = Test()


s_list = ['experience']
search_words = myCall.corpus_synonym(s_list)
print(search_words)

for i in myCall.matched_sentences(search_words,myCall.basic_cleaning(myCall.iterate_data(train_df))):
    #myCall._removeNonAscii(i.encode("utf-8"))
    print(i.encode("ascii", "ignore"))
    with open("D:\\Output.txt", "a") as text_file:
        print("{}".format(i.encode("ascii", "ignore")), file=text_file)
#print(myCall.corpus_synonym(search_words))

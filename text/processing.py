""" Input paragraph for our pre-processing """
my_para = "I bought a Phone today...          The phone is very nice :)      "
 
""" Removing extra white spaces using regular expressions """
import re
my_para = re.sub('\s+', ' ', my_para)
print my_para
# I bought a Phone today... The phone is very nice :) #

 
""" Removing the extra periods using regular expressions """
my_para = re.sub('\.+', '.', my_para)
print my_para
# I bought a Phone today. The phone is very nice :) #

 
""" Removing the special characters using string replace """
special_char_list = [':', ';', '?', '}', ')', '{', '(']
for special_char in special_char_list:
    my_para = my_para.replace(special_char, '')
print my_para
# I bought a Phone today. The phone is very nice #

 
""" Standardizing the text by converting them to lower case """
my_para = my_para.strip().lower()
print my_para
# i bought a phone today. the phone is very nice #

 
""" Import?ing the necessary modules for stopwords removal"""
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
eng_stopwords = stopwords.words('english')     ## eng_stopwords is the list of english stopwords

""" Tokenizing the paragraph first and then removing the stop words """
wordList = word_tokenize(my_para)                                     ## Tokenizing the paragraph
wordList = [word for word in wordList if word not in eng_stopwords]   ## Removing the stopwords
print wordList
# ['bought', 'phone', 'today', '.', 'phone', 'nice'] # 
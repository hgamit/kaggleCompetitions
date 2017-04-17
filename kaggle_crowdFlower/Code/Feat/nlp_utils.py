
"""
__file__

    nlp_utils.py

__description__

    This file provides functions to perform NLP task, e.g., TF-IDF and POS tagging.

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import re
import sys
import nltk
from bs4 import BeautifulSoup
from replacer import CsvWordReplacer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
sys.path.append("../")
from param_config import config


################
## Text Clean ##
################
## synonym replacer
replacer = CsvWordReplacer('%s/synonyms.csv' % config.data_folder)
## other replace dict
## such dict is found by exploring the training data
replace_dict = {
    "nutri system": "nutrisystem",
    "soda stream": "sodastream",
    "playstation's": "ps",
    "playstations": "ps",
    "playstation": "ps",
    "(ps 2)": "ps2",
    "(ps 3)": "ps3",
    "(ps 4)": "ps4",
    "ps 2": "ps2",
    "ps 3": "ps3",
    "ps 4": "ps4",
    "coffeemaker": "coffee maker",
    "k-cups": "k cup",
    "k-cup": "k cup",
    "4-ounce": "4 ounce",
    "8-ounce": "8 ounce",
    "12-ounce": "12 ounce",
    "ounce": "oz",
    "button-down": "button down",
    "doctor who": "dr who",
    "2-drawer": "2 drawer",
    "3-drawer": "3 drawer",
    "in-drawer": "in drawer",
    "hardisk": "hard drive",
    "hard disk": "hard drive",
    "harley-davidson": "harley davidson",
    "harleydavidson": "harley davidson",
    "e-reader": "ereader",
    "levi strauss": "levi",
    "levis": "levi",
    "mac book": "macbook",
    "micro-usb": "micro usb",
    "screen protector for samsung": "screen protector samsung",
    "video games": "videogames",
    "game pad": "gamepad",
    "western digital": "wd",
    "eau de toilette": "perfume",
}

def clean_text(line, drop_html_flag=False):
    names = ["query", "product_title", "product_description"]
    for name in names:
        l = line[name]
        if drop_html_flag:
            l = drop_html(l)
        l = l.lower()
        ## replace gb
        for vol in [16, 32, 64, 128, 500]:
            l = re.sub("%d gb"%vol, "%dgb"%vol, l)
            l = re.sub("%d g"%vol, "%dgb"%vol, l)
            l = re.sub("%dg "%vol, "%dgb "%vol, l)
        ## replace tb
        for vol in [2]:        
            l = re.sub("%d tb"%vol, "%dtb"%vol, l)

        ## replace other words
        for k,v in replace_dict.items():
            l = re.sub(k, v, l)
        l = l.split(" ")

        ## replace synonyms
        l = replacer.replace(l)
        l = " ".join(l)
        line[name] = l
    return line
    

###################
## Drop html tag ##
###################
def drop_html(html):
    return BeautifulSoup(html, "html5lib").get_text(separator=" ")
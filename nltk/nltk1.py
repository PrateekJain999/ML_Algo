# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:46:10 2019

@author: prateek jain
"""

from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize 


#EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkishblue. You shouldn't eat cardboard." 
 
#print(sent_tokenize(EXAMPLE_TEXT))
#print(word_tokenize(EXAMPLE_TEXT))
#print(set(stopwords.words('english')))
#print("\n\n",stopwords.words('english')) 
 
#example_sent = "This is a sample sentence, showing off the stop words filtration." 
# 
#stop_words = set(stopwords.words('english')) 
# 
#word_tokens = word_tokenize(example_sent) 
 
#filtered_sentence = [w for w in word_tokens if not w in stop_words] 
 
#filtered_sentence = [] 
# 
#for w in word_tokens:
#    if w not in stop_words:
#        filtered_sentence.append(w) 
# 
#print(word_tokens)
#print(filtered_sentence) 

ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"] 

for w in example_words:
    print(ps.stem(w))
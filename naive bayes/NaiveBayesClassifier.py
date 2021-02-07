# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:17:49 2019

@author: prateek jain
"""

from nltk.classify import NaiveBayesClassifier

def word_feats(words):
    return dict([(word,True) for word in words])

positive_vocab=['outstanding','fantastic','terrific','good','nice','great',':)','happy']
negative_vocab=['bad','terrible','useless','hate',':(']
neutral_vocab=['movie','the','sound','was','is','actors','did','know','words','not']

positive_features=[(word_feats(pos),'pos') for pos in positive_vocab]
negative_features=[(word_feats(neg),'neg') for neg in negative_vocab]
neutral_features=[(word_feats(neu),'neu') for neu in neutral_vocab]

train_set=negative_features+positive_features+neutral_features

classifier=NaiveBayesClassifier.train(train_set)

neg=0
pos=0
neu=0
sentence=input('enter a string : ')
sentence=sentence.lower()
words=sentence.split(' ')

print(words)

for word in words:
    classResult=classifier.classify(word_feats(word))
    if classResult == 'neg':
        neg=neg+1
    if classResult == 'pos':
        pos=pos+1
    if classResult == 'neu':
        neu=neu+1
        
print('Positive : '+str(float(pos)/len(words)))
print(len(words))
print(pos)
print('Negative : '+str(float(neg)/len(words)))
print(len(words))
print(neg)
print('Netural : '+str(float(neu)/len(words)))
print(len(words))
print(neu)
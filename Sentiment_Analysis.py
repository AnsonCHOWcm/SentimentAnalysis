#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:34:35 2021

@author: ccm
"""
## Reading the Data Set (Review and Labels)

from collections import Counter
import Sentiment 

def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()


## Train the Model

mlp = Sentiment.SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.05,learning_rate=0.01)

mlp.train(reviews[:-1000],labels[:-1000])

## Train the Model

mlp.test(reviews[-1000:],labels[-1000:])

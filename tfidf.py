### This program calculates the TFIDF values 


from __future__ import division, unicode_literals
import math

idf_words = {}
dict_1 = 0

def tf(word_c, n_words):
    return (word_c / n_words)

##### Actual function for TFIDF which calculates the weights correlated to each word
def tfidf(all_dict):
    y = {}
    seq = {}
    global dict_1
    dict_1 = len(all_dict)
    for key in all_dict.keys():
        n_words = sum(all_dict[key].values())
        y[key] = {}
        seq[key] = 0
        for word in all_dict[key]:
            y[key][word] = tf(all_dict[key][word], n_words) * idf(word, all_dict)
            seq[key] += y[key][word] * y[key][word]
    return y, idf_words, seq

#This function calculates the IDF values of each word sent by the function TFIDF
def idf(word, all_dict):
    global idf_words
    if word not in idf_words:
        containing_n = sum(1 for dict in all_dict if word in dict) ### Having that word in dic and summing it
        idf_words[word] = math.log(dict_1 / (1 + containing_n))
    return idf_words[word]
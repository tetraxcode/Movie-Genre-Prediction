from __future__ import division
import os
import tfidf
import math
import operator
from stopwords import stopwords
import re
global genres
def strip_re(word):
    return re.sub(r"^\W+|\W+$", "", word)

y = {}
idf_ex = {}
SQ = {}

#Declaring the same space of genres
genres = ["Comedy", "Romance", "Drama", "Horror", "Sci_Fi"]

def pre_process():
    #This function performs preprocessing on Training Data
    dir = "Data/Train"
    global genres, y, SQ, idf_ex
    #looping through the list of files and finding the text in them and then calculating its tfidf values
    for dirpath, dirs, files in os.walk(dir):
        for filename in files:
            if filename[filename.rfind(".") + 1:] == "txt":
                i = filename
                y[i] = {}
                fname = os.path.join(dirpath, filename)
                f = open(fname, "r", encoding="latin1")
                text = f.read()
                f.close()
                tokens = text.split()
                for token in tokens:
                    token = strip_re(token)
                    if token.lower() in stopwords:
                        continue
                    if token not in y[i]:
                        y[i][token] = 1
                    else:
                        y[i][token] += 1
    y, idf_ex, SQ = tfidf.tfidf(y)

def word_weight(d):
    #Self explanatory: Calculates weight and its values
    global idf_ex, y, SQ
    total_words = sum(d.values())
    dsq = 0
    d1 = {}
    for word in d:
        if word in idf_ex:
            d1[word] = tfidf.tf(d[word], total_words) * idf_ex[word]
            dsq += d1[word] * d1[word]
    return d1, dsq

def c_distance(d, dsq, file):
    #Calculates distances
    global idf_ex, y, SQ
    denom = 0.000001
    for word in d:
        if word in y[file]:
            denom += y[file][word] * d[word]
    num = math.sqrt(SQ[file]) * math.sqrt(dsq)
    dis = num / denom
    return dis

def distance_eu(d, file, total_words):
    #Calculates distances
    global idf_ex, y, SQ
    num = 0
    flag = 0.000001
    for word in d:
        if word in y[file]:
            flag += 1
            num += (y[file][word] - d[word]) * (y[file][word] - d[word])
    if num == 0:
        num = total_words / flag
    else:
        num = num * total_words / flag
    if flag <= 5:
        dis = 999999
    else:
        dis = math.sqrt(num)
    return dis

def getResponse(d, k):
    #Creates a dictionary with distance as values
    global y, genres
    distances = {}
    total_words  = len(d)
    d, dsq = word_weight(d)
    for file in y.keys():
        dist = c_distance(d, dsq, file)
        distances[file] = dist
    distances = sorted(distances.items(), key=operator.itemgetter(1))
    i = 0
    classVotes = {}
    for file in distances:
        neighbor = file[0]
        print(neighbor + str(file[1]))
        for genre in genres:
            if genre in neighbor:
                response = genre
                if response in classVotes:
                    classVotes[response] += 1
                else:
                    classVotes[response] = 1
        i += 1
        if i > k:
            break
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    res = []
    i = 0
    for k in sortedVotes:
        res.append(k[0])
    return res



if __name__ == '__main__':
    #Beginning of the execution
    st = ""
    res = {}
    for genre in genres:
        res[genre] = [[0, 0], [0, 0]]
    dir = "Data/Test"
    #calling preprocess to get the tfidf values
    pre_process()
    for dirpath, dirs, files in os.walk(dir):
        #Iterating through the test files to test them
        for filename in files:
            if filename[filename.rfind(".") + 1:] == "txt":
                d = {}
                fname = os.path.join(dirpath, filename)
                f = open(fname, "r", encoding="latin1")
                text = f.read()
                f.close()
                tokens = text.split()
                for token in tokens:
                    token = strip_re(token)
                    if token.lower() in stopwords:
                        continue
                    if token not in d:
                        d[token] = 1
                    else:
                        d[token] += 1
                k = 5
                result = getResponse(d, k)

                actual_set = set()
                for genre in genres:
                    if genre in fname:
                        actual_set.add(genre)
                result_set = set(result[0:min(len(result), len(actual_set))])
                #print(filename + " - res: " + str(result_set)+ " - actual: "+ str(actual_set))
                for genre in actual_set.intersection(result_set):
                    res[genre][0][0] += 1
                for genre in result_set - actual_set:
                    res[genre][1][0] += 1
                for genre in actual_set - result_set:
                    res[genre][0][1] += 1

    for genre in genres:
        precision = res[genre][0][0] / (res[genre][0][0] + res[genre][1][0])
        recall = res[genre][0][0] / (res[genre][0][0] + res[genre][0][1])
        finalscore = 2 * precision * recall / (precision + recall) 
        print("\n" + genre + "\nprecision: " + str(precision) + "\nrecall: " + str(recall) + "\nscore: " + str(finalscore))
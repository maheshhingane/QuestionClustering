#!/usr/bin/env python

from __future__ import division
import sys
import csv
import shutil
import nltk
from nltk.corpus import stopwords
import random

allwords=[]
queswithword=[]
idf=[]
tfidfarray=[]
tfidf=[]
temp=0
tf=0
i=0
j=0
flag=0
counter=0

#Accept command line arguments
if len(sys.argv) != 3:
    print "USAGE: python <script_name.py> <training_file> <k>"
    sys.exit(0)
else:
    training_file = sys.argv[1]
    k = sys.argv[2]

stop = stopwords.words('english')

#Extract unique tags from the training data
training_data = csv.reader(open(training_file,'rb'))
print "Extracting words from training data..."
for training_row in training_data:
    print "Extracting words. At question ", counter
    counter += 1
    words = [x for x in (training_row[1]+training_row[2]).lower().split() if x not in stop]
    for word in words:
        if word not in allwords:
            allwords.append(word)
            queswithword.append([])
            queswithword[i].append(training_row[0])
            i += 1
        else:
            for j in range(len(allwords)):
                if allwords[j] == word:
                    queswithword[j].append(training_row[0])
print "Words extracted successfully."

#Create the IDF array
print "Generating IDF values for training data..."
for i in range(len(allwords)):
    if len(queswithword[i])>0:
        idf.append(1/(len(queswithword[i])))
    else:
        idf.append(0)
print "IDF values generated successfully."

#Create the TFIDF table
counter=0
training_data = csv.reader(open(training_file,'rb'))
training_data.next()
print "Generating TFIDF values for each word in training data..."
for training_row in training_data:
    print "Generating TFIDF. At question ", counter
    counter += 1
    for word in [x for x in (training_row[1]+training_row[2]).lower().split() if x not in stop]:
        tf = 0
        tf += 5*(training_row[1].lower().count(word))
        tf += training_row[2].lower().count(word)

        for i in range(len(allwords)):
            if allwords[i] == word:
                try:
                    temp = float(idf[i])
                except ValueError:
                    temp = 0
                tfidf.append([training_row[0],word,tf*temp])
                break
print "TFIDF values generated successfully."

#Normalize the TFIDF values
print "Normalizing the question vectors..."
for j in range(counter):
    print "Normalizing question vectors. At question ", j
    factor = 0
    for i in range(len(tfidf)):
        if j == int(tfidf[i][0]):
            factor += int(tfidf[i][2]) ** 2
    factor = factor ** (0.5)
    for i in range(len(tfidf)):
        if j == int(tfidf[i][0]):
            if factor != 0:
                tfidf[i][2] = float(tfidf[i][2]) / factor
print "Question vectors normalized successfully."

#Generate initial centroids
c = 0
centroids=[]
for x in range(int(k)):
    c = random.randint(0,counter-1)
    centroids.append(c)
    print "Centroid %s = Question %s" % (x, c)

#Cluster the questions
distance=[]
for x in range(int(k)):
    distance.append([])
    for i in range(counter):
        print "Calculating distance of question %s from cluster %s" % (i,x)
        d = 0
        for j in range(len(tfidf)):
            if i == int(tfidf[j][0]):
                for p in range(len(tfidf)):
                    if int(tfidf[p][0]) == int(centroids[x]) and tfidf[p][1] == tfidf[j][1]:
                        d += (float(tfidf[j][2]) - float(tfidf[p][2])) ** 2
                        continue
        d = float(d) ** (0.5)
        distance[x].append(d)

cluster=[]
clusterid=-1
for i in range(counter):
    min = 1000
    for x in range(int(k)):
        if distance[x][i] < min:
            min = distance[x][i]
            clusterid = x
    cluster.append(clusterid)

resultfile = open('result.txt','wb')
result = csv.writer(resultfile)
result.writerow(["questionId","clusterId"])
for i in range(len(cluster)):
    result.writerow([i,cluster[i]])

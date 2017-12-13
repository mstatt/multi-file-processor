# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 01:25:12 2017
Complete document analysis:
1) Fuzzy String compare for file similarity
2) Word frequency counter
3) Phrase frequency counter
@author: MStattelman
"""

#Imports
import pandas as pd
import glob
import re
import os
import nltk
import collections
from collections import Counter
from nltk import ngrams
import sys
from math import log
import time
import difflib
import itertools
import uuid
from functools import reduce
from statistics import mean, stdev



#--------------Set up directories and Variables
#Set start time to calculate time of processing
start = time.time()
#Set file extension for specific filetypes
fileext = '.txt'
#Set directory of files for processing
compdir  = 'datafiles/'
#Create a output directory based on a UID
gui = os.path.join(str(uuid.uuid4().hex))
outdir = gui +'/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

#get all of the files in the directory into a list
txt_files = list(filter(lambda x: x.endswith(fileext), os.listdir(compdir)))

def geo_mean_calc(n):
    """
    Calculate the Geomaen
    """
    geomean = lambda n: reduce(lambda x,y: x*y, n) ** (1.0 / len(n))
    return geomean(n)


def compareEach(x,y):
    """
    Compare the 2 files passed in using fuzzy string compare
    """
    with open(compdir + x, 'r') as myfile:
        data=myfile.read().replace('\n', '').lower()
        myfile.close()
    with open(compdir + y, 'r') as myfile2:
        data2=myfile2.read().replace('\n', '').lower() 
        myfile2.close()
    
    return difflib.SequenceMatcher(None, data, data2).ratio()
   
#Set up lists for file names and Fuzzy logic calculations
aList = []
f1 = []
f2 = []
bList = []
#Loop through each list item and compare it against the other items
for a, b in itertools.combinations(txt_files, 2):
    aList.append("File ["+a+"] and file ["+b+"] has a similarity of ");
    f1.append(a)
    f2.append(b)
    bList.append(compareEach(a,b));
    

#Combine both lists into a corolary dictionary
d= dict(zip(aList, bList))

#Save sorted dict as new dictionary from most similar to least
d1 = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

#Save results to file:
fo = open(outdir+'datafile-comparison.txt', "w")
#Print Headers to file
fo.write('File similarity ranked from most to least similar:\n\n')
fo.write('Geometric Mean:'+str(geo_mean_calc(bList))+'\n\n')
fo.write('Arithmatic Mean:'+str(mean(bList))+'\n\n')
#Print Output to file
for k, v in d1.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')
fo.close()



#Use tweet tokenizer to prevent contracted words from spliting
from nltk.tokenize import TweetTokenizer

def remove_punctuation(text):
    # Removes all punctuation and conotation from the string and returns a 'plain' string
    punctuation2 = '-&'+'®©™€â´‚³©¥ã¼•ž®è±äüöž!@#Â“§$%^*()î_+€$=¿{”}[]:«;"»\â¢|<>,.?/~`0123456789'
    for sign in punctuation2:
        text = text.replace(sign, " ")
    return text


#Set length of word combinations for use in counters.
phrase_len = 4
term_len = 1

corpus = []
path = compdir

file_list = []
os.chdir(path)
#Get all files in the directory loaded into the corpus
for file in glob.glob("*.txt"):
    file_list.append(file)
    f = open(file)
    corpus.append(remove_punctuation(f.read()))
    f.close()

frequencies0 = Counter([])
frequencies = Counter([])
#Cycle through corpus to generate frequencies metrics
for text in corpus:
    tknzr = TweetTokenizer()
    token = tknzr.tokenize(text)
    #Frequency for words
    single = ngrams(token, term_len)
    frequencies0 += Counter(single)
    #Frequency for phrases
    quadgrams = ngrams(token, phrase_len)
    frequencies += Counter(quadgrams)

od0 = collections.OrderedDict(frequencies0.most_common())
od = collections.OrderedDict(frequencies.most_common())

#Build dataframes
os.chdir('..')

#Create output for fuzzy string compare as dataframe
dfz = pd.DataFrame(list(zip(f1, f2, bList)),
              columns=['File #1','File #2', 'Similarity'])
dfz.sort_values(["Similarity"], inplace=True, ascending=False)
dfz.index = pd.RangeIndex(len(dfz.index))

#Create output for word frequency dataframe
df0 = pd.DataFrame.from_dict(od0, orient='index').reset_index()
df0 = df0.rename(columns={'index':'Word', 0:'Count'})


#Create output for Phrase frequency as dataframe
df = pd.DataFrame.from_dict(od, orient='index').reset_index()
df = df.rename(columns={'index':'Phrase', 0:'Count'})


#Get a count of all words and phrases
Count_Words=df0.shape[0]
Count_Phrase=df.shape[0]

#Generate html files from dataframes
dfz.to_html(open(outdir +'Sim.html', 'a'))
df0.to_html(open(outdir +'Word.html', 'a'))
df.to_html(open(outdir +'Phrase.html', 'a'))


#Write File list to File
with open (outdir+"complete.txt","a")as fp1:
   fp1.write("Execution time: " + str(time.time() - start) +"s\n\n")
   fp1.write("With a total unique word count of:"+str(Count_Words)+"\n\n")
   fp1.write("With a total unique phrase count of:"+str(Count_Phrase)+"\n\n")
   fp1.write("The following files ("+str(len(file_list))+") were processed in the comparisons:\n\n")
   for line in file_list:
       fp1.write(line+"\n\n")
   fp1.close()

#Generate Analysis pdf form files collection
import pdfkit
pdfkit.from_file([outdir+"complete.txt",outdir+'Sim.html',outdir +'Word.html',outdir +'Phrase.html'], outdir +' Task-'+gui+'-Document-Analysis.pdf')
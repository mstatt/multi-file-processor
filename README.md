# multi-file-processor
This single Python 3.6.3 script consumes all of the .txt files in a specified directory for processing

1) It starts off by using fuzzy logic to calculate the similarities between the files.
2) Then does a word frequency calculation.
3) It does a 4 word phrase frequency calculation.
4) Generates some statistics.
5) Seperates each set of results into distinct html files.
6) Combines the stats and the result sets into a pdf file.

Of note there are libraries that are imported as well as moduals that ineed to be installed:
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

Installs:
pdfkit(Optional, and can be commented out),
Fuzzy, wkhtmltopdf




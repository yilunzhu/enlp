#!/usr/bin/env python3

"""
Converts chaos.html into JSON. A sample of the input:

<xxx1><p>Dearest <i>creature</i> in <i>creation</i><br>
<xxx2>Studying English <i>pronunciation</i>,<br>
<xxx3><tt>&nbsp;&nbsp;&nbsp;</tt>I will teach you in my <i>verse</i><br>
<xxx4><tt>&nbsp;&nbsp;&nbsp;</tt>Sounds like <i>corpse</i>, <i>corps</i>, <i>horse</i> and <i>worse</i>.</p>

A hand-formatted portion of the output (note that indentation, line breaks,
order of dict entries, etc. don't matter as long as the data matches):

[
    {"stanza": 1,
     "lines": [
          {"lineId": "1-1", "lineNum": 1, "text": "Dearest creature in creation",
           "tokens": ["Dearest", "creature", "in", "creation"],
           "rhymeWord": "creation"},
          {"lineId": "1-2", "lineNum": 2, "text": "Studying English pronunciation,",
           "tokens": ["Studying", "English", "pronunciation", ","],
           "rhymeWord": "pronunciation"},
          ...
     ]},
    {"stanza": 2,
     "lines": [
          {"lineId": "2-1", "lineNum": 1, "text": "I will keep you, Susy, busy,",
           "tokens": ["I", "will", "keep", "you", ",", "Susy", ",", "busy", ","],
           "rhymeWord": "busy"},
          ...
     ]},
     ...
]
"""

import json, re, codecs
from nltk import word_tokenize
#import urllib.request

def hasalpha(token):
    for letter in token:
        if re.match("[A-Za-z]", letter) is None:
            return False
    return True
# TODO: whether any character in the token is a letter

# regex that breaks an HTML line into parts: line number within the stanza, main portion, spacing
#LINE_RE = # TODO:

# TODO: read from chaos.html, construct data structure, write to chaos.json


poem_total = []
poem_lines = []
poem_stanza = {}
n = 0
#url="http://ncf.idallen.com/english.html"
#data = urllib.request.urlopen(url).readlines()
f = codecs.open("chaos.html", "r")
data = f.readlines()

poem=[]
for line in data:
    #line = line.decode('utf-8')
    match = re.search(r"<xxx.+", line)
    #print(type(match))
    if match is not None:
        cleaned = re.sub(r"<xxx|>|</?p>|</?i>|<br>|</?tt>|&nbsp;|\n|</?b>", "", line)
        #tokenization = word_tokenize(line)
        #print(tokenization)
        poem_line = {}
        if cleaned[0] == "1":
            n += 1
        cleaned2 = re.sub("-", " - ", cleaned)
        words = word_tokenize(cleaned2[1:])
        poem_line['lineId'] = str(n) + "-" + cleaned[0]
        poem_line['lineNum'] = cleaned[0]
        poem_line['text'] = cleaned[1:]
        poem_line['tokens'] = words
        m = -1
        while not hasalpha(poem_line['tokens'][m]):
            m -= 1
        poem_line['rhymeWord'] = poem_line['tokens'][m]
        poem_lines.append(poem_line)
        if len(poem_lines) == 4:
            poem_stanza['stanza'] = n
            poem_stanza['lines'] = poem_lines
            poem_total.append(poem_stanza)
            poem_stanza = {}
            poem_lines = []
        '''elif cleaned[-3:] == "UP!":
            poem_stanza['stanza'] = n
            poem_stanza['lines'] = poem_lines
            poem_total.append(poem_stanza)'''
    else:
        continue

#print(poem_total)
#print(poem)
python_to_json = json.dumps(poem_total, indent=2)

output = open("data.json", "w")
output.write(python_to_json)
output.close()

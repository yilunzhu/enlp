#!/usr/bin/env python3

import json, re
from collections import defaultdict
from nltk.corpus import cmudict # need to have downloaded the data through NLTK
import copy

def isExactRhyme(p1, p2):
    """
    TODO: Explain your heuristic here.
        If the vowel and the codas (if any) in the last syllable are the same, p1 and p2 have identical rhyme.
        When matching identical rhyme, the stress of vowels are ignored.
    """

    '''n = -1
    new1 = copy.deepcopy(p1)
    new2 = copy.deepcopy(p2)
    for phone in p1[::-1]:
        if re.search("[012]", phone):
            new1[n] = phone[:-1]
            new2[n] = p2[n][:-1]
            for x in range(n, 0):
                if new1[x] != new2[x]:
                    return False
            return True
        else:
            n -= 1'''

    new1 = p1[::-1]
    new2 = p2[::-1]
    new1 = re.search(r'^[\w\s]*?[0-9].*?[AEIOU]', new1)
    new2 = re.search(r'^[\w\s]*?[0-9].*?[AEIOU]', new2)
    new1 = re.sub(r'[012]', '', new1.group())
    new2 = re.sub(r'[012]', '', new2.group())
    return new1 == new2

    #return # TODO: whether pronunciations p1 and p2 rhyme

# testcode
'''creation = ["K","R","IY0","EY1","SH","AH0","N"]
pronunciation = ["P","R","OW0","N","AH2","N","S","IY0","EY1","SH","AH0","N"]
print(isExactRhyme(creation, pronunciation))'''

# Load chaos.pron.json


# For each pair of lines that are supposed to rhyme,
# check whether there are any pronunciations of the words that
# make them rhyme according to cmudict and your heuristic.
# Print the rhyme words with their pronunciations and whether
# they are deemed to rhyme or not
# so you can examine the effects of your rhyme detector.
# Count how many pairs are deemed to rhyme vs. not.

with open("chaos.pron.json", "r") as f:
    file = f.read()
lst = []

json_format = json.loads(file)
counter_total = 0
counter_rhyming = 0
count_identical = 0
for stanza in json_format:
    #for line in stanza['lines']:
    lines = stanza['lines']
    for x in range(0, 4, 2):
        line1 = lines[x]
        line2 = lines[x+1]
        if line1['rhymeProns'] is None or line2['rhymeProns'] is None:
            continue
        else:
            counter_total += 1
            flag = 0
            len1 = len(line1['rhymeProns'])
            len2 = len(line2['rhymeProns'])
            for y in range(0, len1):
                prons1 = line1['rhymeProns'][y]
                count_identicaly = 0
                for z in range(0, len2):
                    prons2 = line2['rhymeProns'][z]
                    if isExactRhyme(prons1, prons2):
                        lst.append(line1['rhymeWord'] + " " + line2['rhymeWord'])                       #fsadsdfsddfssd
                        #print(line1['lineId'] + " " + line2['lineId'])
                        count_identicaly += 1
                        if flag == 1:
                            continue
                        counter_rhyming += 1
                        flag = 1
                    '''else:                                                                                        #fsddfd
                        if [line1['rhymeWord'] + " " + line2['rhymeWord']] not in lst:
                            print(line1['rhymeWord'] + " " + line2['rhymeWord'])'''

                if len2 >= 2:
                    if count_identicaly != 0 and count_identicaly != len2:
                        count_identical += 1
                        print(line1['rhymeWord'] + " " + line2['rhymeWord'])
                        break

            for u in range(0, len2):
                pronsu = line2['rhymeProns'][u]
                count_identicalx = 0
                for w in range(0, len1):
                    pronsw = prons1 = line1['rhymeProns'][w]
                    if isExactRhyme(pronsw, pronsu):
                        count_identicalx += 1
                if len1 >= 2:
                    if count_identicalx != 0 and count_identicalx != len1:
                        count_identical += 1
                        print(line1['rhymeWord'] + " " + line2['rhymeWord'])
                        break


print("rhyming lines doing disambiguity: %d" % count_identical)
print("pairs having identical rhymes: %d" % counter_rhyming)
print("pairs having phonetic representation: %d" % counter_total)
print(float(counter_rhyming) / counter_total)




"""
TODO: Answer the questions briefly:

- How many pairs of lines that are supposed to rhyme actually have rhyming pronunciations
according to your heuristic and cmudict?

68.
...

- For how many lines does having the rhyming line help you disambiguate
between multiple possible pronunciations?

13.
...

- What are some reasons that lines supposed to rhyme do not,
according to your rhyme detector? Give examples.

Reason 1: Vowels are different, but probably stressed vowels, rather than unstressed vowels, are the pivot 
in the judgement of rhyming
4-1: via        ["V", "AY1", "AH0"], ["V", "IY1", "AH0"]
4-2: choir      ["K", "W", "AY1", "ER0"]

Reason 2: Codas are different, and they do not play an essential role in rhyming
10-3: broad     ["B", "R", "AO1", "D"]
10-4: reward    ["R", "IH0", "W", "AO1", "R", "D"], ["R", "IY0", "W", "AO1", "R", "D"]

"""
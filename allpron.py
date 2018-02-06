#!/usr/bin/env python3

import json, re
from collections import defaultdict
from nltk import word_tokenize
from nltk.corpus import cmudict # need to have downloaded the data through NLTK

""" Sample part of the output (hand-formatted):

{"lineId": "1-1", "lineNum": 1,
 "text": "Dearest creature in creation",
 "tokens": ["Dearest", "creature", "in", "creation"],
 "rhymeWord": "creation",
 "rhymeProns": ["K R IY0 EY1 SH AH0 N"]},

{"lineId": "4-1", "lineNum": 1,
 "text": "Previous, precious, fuchsia, via",
 "tokens": ["Previous", ",", "precious", ",", "fuchsia", ",", "via"],
 "rhymeWord": "via",
 "rhymeProns": ["V AY1 AH0", "V IY1 AH0"]
},
"""

# Load the cmudict entries into a data structure.
# Store each pronunciation as a STRING of phonemes (separated by spaces).
...

# Load chaos.json
...

# For each line of the poem, add a "rhymeProns" entry
# which is a list parallel with "rhymeWords".
# For each word, it contains a list of possible pronunciations.
...

# Write the enhanced data to chaos.pron.json
...


arphabet = cmudict.dict()

with open("data.json", "r") as f:
    file = f.read()

json_format = json.loads(file)
counter = 0
oov = []
for stanza in json_format:
    for line in stanza['lines']:
        #print(line['rhymeWord'])
        try:
            word_list = []
            for pro in arphabet[line['rhymeWord'].lower()]:
                letter_list = ""
                for pho in pro:
                    if pho != pro[-1]:
                        letter_list += pho + " "
                    else:
                        letter_list += pho
                word_list.append(letter_list)
            line['rhymeProns'] = word_list
        except:
            line['rhymeProns'] = None
            oov.append(line['rhymeWord'])
            counter += 1
print(counter)
print(oov)
python_to_json = json.dumps(json_format, indent=2)
output = open("chaos.pron.json", "w")
output.write(python_to_json)
output.close()


"""
TODO: Answer the question:

- How many rhyme words are NOT found in cmudict (they are "out-of-vocabulary", or "OOV")?
Give some examples.

17.
['ague', 'Terpsichore', 'reviles', 'endeavoured', 'tortious', 'clamour', 'clangour', 'hygienic', 'inveigle', 
'mezzotint', 'Cholmondeley', 'antipodes', 'obsequies', 'dumbly', 'vapour', 'fivers', 'gunwale']



"""

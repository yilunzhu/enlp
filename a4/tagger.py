#!/usr/bin/env python3
"""
ENLP A4: HMM for Part-of-Speech Tagging

Usage: 
  python tagger.py baseline
  python tagger.py hmm

(Nathan Schneider; adapted from Richard Johansson)
"""
from math import log
from collections import Counter, defaultdict

import sys, os, time, platform, nltk

# utility functions to read the corpus

def read_tagged_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence

def read_tagged_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_tagged_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_tagged_sentence(f)
    return sentences


# utility function for color-coding in terminal
# https://gist.github.com/ssbarnea/1316877
def accepts_colors(handle=sys.stdout):
    if (hasattr(handle, "isatty") and handle.isatty()) or \
        ('TERM' in os.environ and os.environ['TERM']=='ANSI'):
        if platform.system()=='Windows' and not ('TERM' in os.environ and os.environ['TERM']=='ANSI'):
            return False #handle.write("Windows console, no ANSI support.\n")
        else:
            return True
    else:
        return False


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

ALPHA = .1
allTagCounts = Counter()
tag_lst = [START, END]
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because missing Counter entries default to 0, not log(0)
transitionDists = {}
emissionDists = {}

def learn(tagged_sentences):
    """
    Record the overall tag counts (allTagCounts) and counts for each word (perWordTagCounts) for baseline tagger.
    (These should not have pseudocounts and should only apply to observed words/tags, not START, END, or UNK.)
    
    Learn the parameters of an HMM with add-ALPHA smoothing (ALPHA = 0.1):
     - Store counts + pseudocounts of observed transitions (transitionCounts) and emissions (emissionCounts) for bigram HMM tagger. 
     - Also store a pseudocount for UNK for each distribution.
     - Normalize the distributions and store (natural) log probabilities in transitionDists and emissionDists.
    """

    # store training data counts in allTagCounts, perWordTagCounts, transitionCounts, emissionCounts
    for sent in tagged_sentences:
        prev_tag = START
        for index, (word, tag) in enumerate(sent):
            if tag not in tag_lst:
                tag_lst.append(tag)
            # compute allTagCounts
            allTagCounts[tag] += 1

            # compute perWordTagCounts
            if word not in perWordTagCounts.keys():
                perWordTagCounts[word] = Counter()
                perWordTagCounts[word][tag] += 1
            else:
                perWordTagCounts[word][tag] += 1

            # compute transitionCounts
            if prev_tag not in transitionCounts.keys():
                transitionCounts[prev_tag] = Counter()
                transitionCounts[prev_tag][tag] += 1
            else:
                transitionCounts[prev_tag][tag] += 1
            if (index + 1) == len(sent):
                end_tag = END
                if tag not in transitionCounts.keys():
                    transitionCounts[tag] = Counter()
                    transitionCounts[tag][end_tag] += 1
                else:
                    transitionCounts[tag][end_tag] += 1

            # compute emissionCounts
            if tag not in emissionCounts.keys():
                emissionCounts[tag] = Counter()
            emissionCounts[tag][word] += 1
            prev_tag = tag

    # add pseudocounts in transitionCounts and emissionCounts, including for UNK
    for tag_1 in transitionCounts.keys():
        transitionCounts[tag_1][UNK] = 0
        for tag_2 in transitionCounts[tag_1].keys():

            # tag combination that never occurs
            # if tag_2 == START: continue
            # if tag_1 == END: continue
            # if tag_1 == START and tag_2 == END: continue
            # if tag_1 not in transitionCounts.keys():
            #     transitionCounts[tag_1] = Counter()
            transitionCounts[tag_1][tag_2] += ALPHA


    for key in emissionCounts.keys():
        emissionCounts[key][UNK] = 0
        for v in emissionCounts[key].keys():
            emissionCounts[key][v] += ALPHA

    # normalize counts and store log probability distributions in transitionDists
    for previous in transitionCounts.keys():
        total = 0
        for comb in transitionCounts[previous].keys():
            total += transitionCounts[previous][comb]
        for tag in transitionCounts[previous].keys():
            if previous not in transitionDists.keys():
                transitionDists[previous] = {}
            transitionDists[previous][tag] = log(float(transitionCounts[previous][tag]) / total)

    # normalize counts and store log probability distributions in emissionDists
    for tag in emissionCounts.keys():
        total = 0
        for word in emissionCounts[tag].keys():
            total += emissionCounts[tag][word]
        for word in emissionCounts[tag].keys():
            if tag not in emissionDists.keys():
                emissionDists[tag] = {}
            emissionDists[tag][word] = log(float(emissionCounts[tag][word]) / total)


def baseline_tag_sentence(sentence):
    """
    Tag the sentence with a most-frequent-tag baseline: 
    For each word, if it has been seen in the training data, 
    choose the tag it was seen most often with; 
    otherwise, choose the overall most frequent tag in the training data.
    Hint: Use the most_common() method of the Counter class.
    
    Do NOT modify the contents of 'sentence'.
    Return a list of (word, predicted_tag) pairs.
    """
    pred_lst = []
    for word, tag in sentence:
        if word in perWordTagCounts.keys():
            pred_tag = max(perWordTagCounts[word], key=perWordTagCounts[word].get)
            pred_lst.append((word, pred_tag))
        else:
            pred_tag = allTagCounts.most_common(1)[0][0]
            pred_lst.append((word, pred_tag))

    return pred_lst

def hmm_tag_sentence(sentence):
    """
    Tag the sentence with the bigram HMM using the Viterbi algorithm.
    Do NOT modify the contents of 'sentence'.
    Return a list of (word, predicted_tag) pairs.
    """
    # fill in the Viterbi chart
    end_item = viterbi(sentence)

    # then retrace your steps from the best way to end the sentence, following backpointers
    lst = retrace(end_item, len(sentence))
    tagged = []
    for index, (word, tag) in enumerate(sentence):
        tup = (word, lst[index])
        tagged.append(tup)
    
    # finally return the list of tagged words
    return tagged


def viterbi(sentence):
    """
    Creates the Viterbi chart, column by column. 
    Each column is a list of tuples representing cells.
    Each cell ("item") holds: the tag being scored at the current position; 
    a reference to the corresponding best item from the previous position; 
    and a log probability. 
    This function returns the END item, from which it is possible to 
    trace back to the beginning of the sentence.
    """
    # make a dummy item with a START tag, no predecessor, and log probability 0
    # current list = [ the dummy item ]
    current = [(START, "", 0)]

    # for each word in the sentence:
    #    previous list = current list
    #    current list = []        
    #    determine the possible tags for this word
    #  
    #    for each tag of the possible tags:
    #         add the highest-scoring item with this tag to the current list
    for index, (word, tag) in enumerate(sentence):
        prev_lst = current
        current = []
        tagset = []
        if word in perWordTagCounts.keys():
            tagset = [t for t in perWordTagCounts[word].keys()]
        else:
            tagset = tag_lst[2:]
        for poss_tag in tagset:
            # if poss_tag == START:
            #     continue
            best_item = find_best_item(word, poss_tag, prev_lst)
            current.append(best_item)

        if (index + 1) == len(sentence):
            max_prob = -10000
            prev_tag = ""
            best_predecessor = ()
            for tup in current:
                prev_tag = tup[0]
                if END in transitionDists[prev_tag].keys():
                    prob = tup[2] + transitionDists[prev_tag][END]
                else:
                    prob = tup[2] + transitionDists[prev_tag][UNK]
                if prob > max_prob:
                    max_prob = prob
                    best_predecessor = tup
            # end the sequence with a dummy: the highest-scoring item with the tag END
            return (END, best_predecessor, max_prob)

    
def find_best_item(word, tag, possible_predecessors):    
    # determine the emission probability: 
    #  the probability that this tag will emit this word
    if word in perWordTagCounts.keys():
        emission_prob = emissionDists[tag][word]
    else:
        emission_prob = emissionDists[tag][UNK]
    
    # find the predecessor that gives the highest total log probability,
    #  where the total log probability is the sum of
    #    1) the log probability of the emission,
    #    2) the log probability of the transition from the tag of the 
    #       predecessor to the current tag,
    #    3) the total log probability of the predecessor
    best_prob = -10000.0
    best_predecessor = ()
    for tup in possible_predecessors:
        prev_tag = tup[0]
        if tag in transitionDists[prev_tag].keys():
            prob = emission_prob + transitionDists[prev_tag][tag] + tup[2]
        else:
            prob = emission_prob + transitionDists[prev_tag][UNK] + tup[2]
        if prob > best_prob:
            best_prob = prob
            best_predecessor = tup
    
    # return a new item (tag, best predecessor, best total log probability)
    return (tag, best_predecessor, best_prob)


def retrace(end_item, sentence_length):
    # tags = []
    # item = predecessor of end_item
    # while the tag of the item isn't START:
    #     add the tag of item to tags
    #     item = predecessor of item
    # reverse the list of tags and return it

    tags = []
    item = end_item[1]
    for iter in range(sentence_length):
        tags.append(item[0])
        item = item[1]
    tags.reverse()
    return tags

def joint_prob(sentence):
    """Compute the joint probability of the given words and tags under the HMM model."""
    p = 0   # joint log prob. of words and tags
    prev_tag = START
    for index, (word, tag) in enumerate(sentence):
        if word not in emissionDists[tag].keys():
            if tag in transitionDists[prev_tag].keys():
                p += (transitionDists[prev_tag][tag] + emissionDists[tag][UNK])
            else:
                p += (transitionDists[prev_tag][UNK] + emissionDists[tag][UNK])
        else:
            if tag in transitionDists[prev_tag].keys():
                p += (transitionDists[prev_tag][tag] + emissionDists[tag][word])
            else:
                p += (transitionDists[prev_tag][UNK] + emissionDists[tag][word])
        prev_tag = tag
        if (index + 1) == len(sentence):
            if END in transitionDists[tag].keys():
                p += transitionDists[tag][END]
            else:
                p += transitionDists[tag][UNK]
    # assert isfinite(p) and p<0  # Should be negative
    return p


def count_correct(gold_sentence, pred_sentence):
    """Given a gold-tagged sentence and the same sentence with predicted tags,
    return the number of tokens that were tagged correctly overall, 
    the number of OOV tokens tagged correctly, 
    and the total number of OOV tokens."""
    assert len(gold_sentence)==len(pred_sentence)
    correct = 0
    correctOOV = 0
    OOV = 0
    for index, (word, gold_tag) in enumerate(gold_sentence):
        if gold_tag == pred_sentence[index][1]:
            correct += 1
        # compute the number of wrong tags
        else:
            wrong[gold_tag] += 1
        if word not in perWordTagCounts.keys():
            OOV += 1
            if gold_tag == pred_sentence[index][1]:
                correctOOV += 1
    return correct, correctOOV, OOV


# universal tags
# TRAIN_DATA = 'en-ud-train.upos.tsv'
# TEST_DATA = 'en-ud-test.upos.tsv'

# penn tags
TRAIN_DATA = 'en-ud-train.ppos.tsv'
TEST_DATA = 'en-ud-test.ppos.tsv'


train_sentences = read_tagged_corpus(TRAIN_DATA)


# train the bigram HMM tagger & baseline tagger in one fell swoop
trainingStart = time.time()
learn(train_sentences)
trainingStop = time.time()
trainingTime = trainingStop - trainingStart


# decide which tagger to evaluate
if len(sys.argv)<=1:
    assert False,"Specify which tagger to evaluate: 'baseline' or 'hmm'"
if sys.argv[1]=='baseline':
    tagger = baseline_tag_sentence
elif sys.argv[1]=='hmm':
    tagger = hmm_tag_sentence
else:
    assert False,'Invalid command line argument'



if accepts_colors():
    class bcolors:  # terminal colors
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
else:
    class bcolors:
        HEADER = ''
        OKBLUE = ''
        OKGREEN = ''
        WARNING = ''
        FAIL = ''
        ENDC = ''
        BOLD = ''
        UNDERLINE = ''


def render_gold_tag(x):
    (w,gold),(w,pred) = x
    return w + '/' + (bcolors.WARNING + gold + bcolors.ENDC if gold!=pred else gold)
    
def render_pred_tag(x):
    (w,gold),(w,pred) = x
    return w + '/' + (bcolors.FAIL + pred + bcolors.ENDC if gold!=pred else pred)


# word type: wrong tagging
wrong = defaultdict(int)


test_sentences = read_tagged_corpus(TEST_DATA)

nTokens = nCorrect = nOOV = nCorrectOOV = nPerfectSents = nPGoldGreater = nPPredGreater = 0

taggingTime = 0

for sent in test_sentences:
    taggerStart = time.time()
    pred_tagging = tagger(sent)
    taggerStop = time.time()
    taggingTime += taggerStop - taggerStart
    nCorrectThisSent, nCorrectOOVThisSent, nOOVThisSent = count_correct(sent, pred_tagging)
    
    acc = nCorrectThisSent/len(sent)
    
    pHMMGold = joint_prob(sent)
    pHMMPred = joint_prob(pred_tagging)
    print(pHMMGold, ' '.join(map(render_gold_tag, zip(sent,pred_tagging))))
    print(pHMMPred, ' '.join(map(render_pred_tag, zip(sent,pred_tagging))), '{:.0%}'.format(acc))
    
    if pHMMGold > pHMMPred:
        nPGoldGreater += 1
        # assert False
    elif pHMMGold < pHMMPred:
        nPPredGreater += 1
    
    nCorrect += nCorrectThisSent
    nCorrectOOV += nCorrectOOVThisSent
    nOOV += nOOVThisSent
    nTokens += len(sent)
    if pred_tagging==sent:
        nPerfectSents += 1

print('TAGGING ACCURACY BY TOKEN: {}/{} = {:.1%}   OOV TOKENS: {}/{} = {:.1%}   PERFECT SENTENCES: {}/{} = {:.1%}   #P_HMM(GOLD)>P_HMM(PRED): {}   #P_HMM(GOLD)<P_HMM(PRED): {}'.format(nCorrect, nTokens, nCorrect/nTokens, 
            nCorrectOOV, nOOV, nCorrectOOV/nOOV,
            nPerfectSents, len(test_sentences), nPerfectSents/len(test_sentences), 
            nPGoldGreater, nPPredGreater))
print('RUNTIME: TRAINING = {:.2}s, TAGGING = {:.2}s'.format(trainingTime, taggingTime))
# print("wrong tagged: %s" %max(wrong, key=wrong.get))
print(wrong)

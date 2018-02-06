import codecs
import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import re


vocab = codecs.open("brown_vocab_100.txt", "r", encoding="utf-16")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word_index_dict[line.strip()] = i

vocab.close()
f = codecs.open("brown_100.txt", encoding = "utf-16")


#TODO: initialize numpy 0s array
length = len(word_index_dict)
counts = np.zeros(shape=(length, length))


#TODO: iterate through file and update counts
file = [[word.lower() for word in line.split()] for line in f]
f.close()
file[0] = file[0][1:]

previous_word = "<s>"
for line in file:
    for word in line:
        counts[word_index_dict[previous_word], word_index_dict[word]] += 1
        previous_word = word

#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)

#TODO: writeout bigram probabilities
prob1 = str(probs[word_index_dict["all"], word_index_dict["the"]])
prob2 = str(probs[word_index_dict["the"], word_index_dict["jury"]])
prob3 = str(probs[word_index_dict["the"], word_index_dict["campaign"]])
prob4 = str(probs[word_index_dict["anonymous"], word_index_dict["calls"]])

fw = open("bigram_probs.txt", "w")
fw.write("p(the|all) = %s\np(jury|the) = %s\np(campaign|the) = %s\np(calls|anonymous) = %s"
         % (prob1, prob2, prob3, prob4))
fw.close()


# problem 6: Calculating sentence probabilities
sentence = codecs.open("toy_corpus.txt", "r", "utf-16")
output = open("bigram_eval.txt", "w")
m = 0
for line in sentence:
    previous_word = "<s>"
    line = line.lstrip("<s>")
    m += 1
    sentprob = 1
    n = 1
    for word in line.split():
        word = word.lower()
        sentprob *= probs[word_index_dict[previous_word], word_index_dict[word]]
        n += 1
        previous_word = word
    #sentence_probs.append(sentprob)
    perplexity = 1 / pow(sentprob, 1.0 / n)
    output.write("The probability of sentence %d is: %e\nThe perplexity of sentence %d is: %e \n"
                 % (m, sentprob, m, perplexity))
    #print(sentprob)
    #print(perplexity)

sentence.close()
output.close()

generation = open("bigram_generation.txt", "w")
for x in range(0, 10):
    generation.write(GENERATE(word_index_dict, probs, "bigram", 15, "<s>") + "\n")
generation.close()

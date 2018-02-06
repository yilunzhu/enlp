import codecs
import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random


vocab = codecs.open("brown_vocab_100.txt", "r", encoding="utf-16")

word_index_dict = {}
for i, line in enumerate(vocab):
    word_index_dict[line.strip()] = i

f = codecs.open("brown_100.txt", encoding="utf-16")
length = len(word_index_dict)
counts = np.zeros(shape=(length, length))
file = [[word.lower() for word in line.split()] for line in f]
f.close()
file[0] = file[0][1:]

#bigram
previous_word = "<s>"
for line in file:
    for word in line:
        counts[word_index_dict[previous_word], word_index_dict[word]] += 1
        previous_word = word


#trigram
trigram_count = np.zeros((6, len(word_index_dict)))
file[0] = file[0][1:]
previous_word1 = "s"
previous_word2 = "the"
for line in file:
    for word in line:
        if previous_word1 == "in" and previous_word2 == "the":
            trigram_count[0, word_index_dict[word]] += 1
        if previous_word1 == "in" and previous_word2 == "the":
            trigram_count[1, word_index_dict[word]] += 1
        if previous_word1 == "the" and previous_word2 == "jury":
            trigram_count[2, word_index_dict[word]] += 1
        if previous_word1 == "the" and previous_word2 == "jury":
            trigram_count[3, word_index_dict[word]] += 1
        if previous_word1 == "jury" and previous_word2 == "said":
            trigram_count[4, word_index_dict[word]] += 1
        if previous_word1 == "agriculture" and previous_word2 == "teacher":
            trigram_count[5, word_index_dict[word]] += 1
        previous_word1 = previous_word2
        previous_word2 = word

#unsmoothed
prob1 = trigram_count[0, word_index_dict["past"]] / counts[word_index_dict["in"], word_index_dict["the"]]
prob2 = trigram_count[1, word_index_dict["time"]] / counts[word_index_dict["in"], word_index_dict["the"]]
prob3 = trigram_count[2, word_index_dict["said"]] / counts[word_index_dict["the"], word_index_dict["jury"]]
prob4 = trigram_count[3, word_index_dict["recommended"]] / counts[word_index_dict["the"], word_index_dict["jury"]]
prob5 = trigram_count[4, word_index_dict["that"]] / counts[word_index_dict["jury"], word_index_dict["said"]]
prob6 = trigram_count[5, word_index_dict[","]] / counts[word_index_dict["agriculture"], word_index_dict["teacher"]]
print("Unsmoothed probability:")
print("p(past|in, the) = %f" % prob1)
print("p(time|in, the) = %f" % prob2)
print("p(said|the, jury) = %f" % prob3)
print("p(recommended|the, jury) = %f" % prob4)
print("p(that|jury, said) = %f" % prob5)
print("p(,|agriculture, teacher) = %f\n" % prob6)

#smoothed
trigram_count += 0.1
prob_smoothed1 = trigram_count[0, word_index_dict["past"]] / np.sum(trigram_count[0])
prob_smoothed2 = trigram_count[1, word_index_dict["time"]] / np.sum(trigram_count[1])
prob_smoothed3 = trigram_count[2, word_index_dict["said"]] / np.sum(trigram_count[2])
prob_smoothed4 = trigram_count[3, word_index_dict["recommended"]] / np.sum(trigram_count[3])
prob_smoothed5 = trigram_count[4, word_index_dict["that"]] / np.sum(trigram_count[4])
prob_smoothed6 = trigram_count[5, word_index_dict[","]] / np.sum(trigram_count[5])
print("Smoothed probability:")
print("p(past|in, the) = %f" % prob_smoothed1)
print("p(time|in, the) = %f" % prob_smoothed2)
print("p(said|the, jury) = %f" % prob_smoothed3)
print("p(recommended|the, jury) = %f" % prob_smoothed4)
print("p(that|jury, said) = %f" % prob_smoothed5)
print("p(,|agriculture, teacher) = %f" % prob_smoothed6)

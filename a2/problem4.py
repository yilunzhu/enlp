import codecs
import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random


vocab = codecs.open("brown_vocab_100.txt", "r", encoding="utf-16")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word_index_dict[line.strip()] = i

vocab.close()
f = codecs.open("brown_100.txt", encoding="utf-16")


#TODO: initialize numpy 0s array
length = len(word_index_dict)
counts = np.zeros(shape=(length, length))


#TODO: iterate through file and update counts
file = [[word.lower() for word in line.split()] for line in f]
f.close()
file[0] = file[0][1:]

previous_word = "<s>"
for line in file:
    line = line
    for word in line:
        counts[word_index_dict[previous_word], word_index_dict[word]] += 1
        previous_word = word

#TODO: normalize counts
counts += 0.1
probs = normalize(counts, norm='l1', axis=1)

#TODO: writeout bigram probabilities
prob1 = str(probs[word_index_dict["all"], word_index_dict["the"]])
prob2 = str(probs[word_index_dict["the"], word_index_dict["jury"]])
prob3 = str(probs[word_index_dict["the"], word_index_dict["campaign"]])
prob4 = str(probs[word_index_dict["anonymous"], word_index_dict["calls"]])

fw = open("smooth_probs.txt", "w")
fw.write("p(the|all) = %s\np(jury|the) = %s\np(campaign|the) = %s\np(calls|anonymous) = %s"
         % (prob1, prob2, prob3, prob4))
fw.close()


"""
Q: Why did all four probabilities go down in the smoothed model?

Smoothing increases the sum of each row, so that the denominator is bigger than the initial. Therefore, the probability
goes down.


Now note that the probabilities did not all decrease by the same amount. In particular, the two probabilities 
conditioned on ‘the’ dropped only slightly, while the other two probabilities (conditioned on ‘all’ and ‘anonymous’) 
dropped rather dramatically. 
Q: Why did add-α smoothing cause probabilities conditioned on ‘the’ to fall much less than these others?Why did 
add-α smoothing cause probabilities conditioned on ‘the’ to fall much less than these others? And why is this behavior 
(causing probabilities conditioned on ‘the’ to fall less than the others) a good thing?

The training data is small, so that it's easily to go back to the corpus to see what's going on. "all the" occurs once
in the corpus, and the initial sum of counts without smoothing is 1. In this case, add-α smoothing will largely increase
the sum of counts and give p(the|all) a very small probability. So does "anonymous calls." However, many words may occur
given "the", thus smoothing does not decrease the probability too much.

Sometimes though the word combinations get high probability, it may occur very few times in the corpus. The sparse data
leads to high probability, and it's not reasonable to assume that the probability is reliable in a larger corpus. In
this case, smoothing will give sparse data a low probability to not decrease the performance of the n-gram model.
"""


# problem 6: Calculating sentence probabilities
sentence = codecs.open("toy_corpus.txt", "r", "utf-16")
output = open("smoothed_eval.txt", "w")
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

generation = open("smoothed_generation.txt", "w")
for x in range(0, 10):
    generation.write(GENERATE(word_index_dict, probs, "bigram", 15, "<s>") + "\n")
generation.close()

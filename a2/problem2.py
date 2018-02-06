import codecs
import numpy as np
from generate import GENERATE


vocab = codecs.open("brown_vocab_100.txt", "r", encoding="utf-16")

#load the indices dictionary
word_index_dict = {}

for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word_index_dict[line.strip()] = i

vocab.close()
f = codecs.open("brown_100.txt", encoding="utf-16")

#TODO: initialize counts to a zero vector
counts = np.zeros(len(word_index_dict))

#TODO: iterate through file and update counts
for line in f:
    for word in line.split():
        counts[word_index_dict[word.lower()]] += 1
f.close()

#print(np.sum(counts))
#print(counts)

#TODO: normalize and writeout counts.
probs = counts / np.sum(counts)

fw = open("unigram_probs.txt", "w")
fw.write(str(probs))
fw.close()
#print(probs)

"""
Q: Estimate (just by eyeballing) the proportion of the word types that occurred only once in this corpus. 
Do you think the proportion of words that occur only once would be higher or lower if we used a larger corpus 
(e.g., all 57000 sentences in Brown)? Why or why not?

The estimated proportion is around 60%. The proportion will decrease. When you introduce more data and suppose the 
proportion of words that occur only once in the data is around 60%, words occur once in the new corpus (original data
+ introduced data) would decrease, because words occur once in the original data would also appear in the introduced 
data. This would decrease the proportion of words that occur only once.
"""


# problem 6: Calculating sentence probabilities
sentence = codecs.open("toy_corpus.txt", "r", "utf-16")
output = open("unigram_eval.txt", "w")
m = 0
for line in sentence:
    m += 1
    sentprob = 1
    n = 0
    for word in line.split():
        word = word.lower()
        sentprob *= probs[word_index_dict[word]]
        n += 1
    #sentence_probs.append(sentprob)
    perplexity = 1 / pow(sentprob, 1.0 / n)
    output.write("The probability of sentence %d is: %e\nThe perplexity of sentence %d is: %e \n"
                 % (m, sentprob, m, perplexity))
    #print(sentprob)
    #print(perplexity)

sentence.close()
output.close()


generation = open("unigram_generation.txt", "w")
for x in range(0, 10):
    generation.write(GENERATE(word_index_dict, probs, "unigram", 15, "<s>") + "\n")
generation.close()

#!/usr/bin/env python3
"""
ANLP A4: Perceptron

Usage: python perceptron.py NITERATIONS

(Adapted from Alan Ritter)
"""
import sys, os, glob

from collections import Counter, defaultdict
from math import log
from numpy import mean
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer

from evaluation import Eval

def load_docs(direc, lemmatize, labelMapFile='labels.csv'):
    """Return a list of word-token-lists, one per document.
    Words are optionally lemmatized with WordNet."""


    labelMap = {}   # docID => gold label, loaded from mapping file
    with open(os.path.join(direc, labelMapFile)) as inF:
        for ln in inF:
            docid, label = ln.strip().split(',')
            assert docid not in labelMap
            labelMap[docid] = label

    # create parallel lists of documents and labels
    docs, labels = [], []
    for file_path in glob.glob(os.path.join(direc, '*.txt')):
        filename = os.path.basename(file_path)
        # open the file at file_path, construct a list of its word tokens,
        # and append that list to 'docs'.
        # look up the document's label and append it to 'labels'.
        # ...
        with open(file_path) as file:

            # docs.append([word for word in file.read().split()])

            # feature engineering: lowercase
            docs.append([word.lower() for word in file.read().split()])

            labels.append(labelMap[filename])
    return docs, labels

def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()

    # feature engineering: unigram
    for word in doc:
        if word in ff:
            continue
        ff[word] = 1


    # feature: engineering: bigram
    doc = ["<s>"] + doc
    doc += ["</s>"]
    prev_word = doc[0]
    for word in doc:
        if word == doc[0]:
            continue
        ff[(prev_word, word)] = 1
        prev_word = word

    # add bias feature
    ff["__bias"] = 1

    return ff


def load_featurized_docs(datasplit):
    rawdocs, labels = load_docs(datasplit, lemmatize=False)
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in rawdocs:
        featdocs.append(extract_feats(d))
    return featdocs, labels

class Perceptron:
    def __init__(self, train_docs, train_labels, MAX_ITERATIONS=100, dev_docs=None, dev_labels=None):
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        self.weights = {l: Counter() for l in self.CLASSES}
        self.learn(train_docs, train_labels)


    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {l: Counter(c) for l,c in self.weights.items()}


    # problem 4: matrix
    def error_analysis(self, test_docs, test_labels):
        # matrix of 11 languages
        lang_dict = {}
        for index, lang in enumerate(self.CLASSES):
            lang_dict[lang] = index
        lang_matrix = np.zeros(shape=(11, 11))
        for index, dict in enumerate(test_docs):
            # pred_label = self.predict(dict)
            lang_matrix[lang_dict[self.predict(dict)], lang_dict[test_labels[index]]] += 1

        # 10 highest-weighted features & 10 lowest-weighted features
        for lang in lang_dict.keys():
            b = sorted(self.weights[lang].items(), key=lambda x: x[1], reverse=True)
            most10 = []
            m = 0
            for key, value in b:
                if m >= 10:
                    break
                most10.append([key, value])
                m += 1
            least10 = []
            n = 0
            c = sorted(self.weights[lang].items(), key=lambda x: x[1], reverse=False)
            for key, value in c:
                if n >= 10:
                    break
                least10.append([key, value])
                n += 1
            print(lang, file=sys.stderr)
            print("the 10 highest-weighted features: ", most10, "; the 10 lowest-weighted features: ", least10, "\n",
                  file=sys.stderr)

            # precision, recall, F1
            precision = lang_matrix[lang_dict[lang], lang_dict[lang]] / sum(lang_matrix[:, lang_dict[lang]])
            recall = lang_matrix[lang_dict[lang], lang_dict[lang]] / sum(lang_matrix[lang_dict[lang], :])
            f1 = (2 * precision * recall) / (precision + recall)
            print("precision=", precision, ", recall=", recall, ", F1=", f1, "\n", file=sys.stderr)

        return lang_matrix


    def learn(self, train_docs, train_labels):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """
        dev_acc = 0
        for iter in range(self.MAX_ITERATIONS):
            prev_acc = dev_acc
            updates = 0
            for index, dict in enumerate(train_docs):
                gold_label = train_labels[index]
                pred_label = self.predict(dict)
                if pred_label != gold_label:
                    updates += 1
                    # for word in dict.keys():
                    self.weights[pred_label].subtract(dict)
                    self.weights[gold_label].update(dict)


            # compute the number of parameters
            params = 0
            for language in self.weights.keys():
                    params += len(self.weights[language])
            train_acc = self.test_eval(train_docs, train_labels)
            dev_acc = self.test_eval(self.dev_docs, self.dev_labels)
            print("iteration: ", iter, "updates=", updates, "trainAcc=", train_acc,
                  ", devAcc=", dev_acc, "params=", params, file=sys.stderr)

            # best devAcc
            if dev_acc > prev_acc:
                self.copy_weights()


            # updates and stop
            if updates == 0:
                print('Stop at', iter, 'iterations', file=sys.stderr)
                self.weights = self.copy_weights()
                break


    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """
        s = 0
        for word in doc:
            s += doc[word] * self.weights[label][word]
        return s


    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        pred_score = defaultdict(int)

        # compute the testAcc when the devAcc is the highest
        for language in self.weights.keys():
            pred_score[language] = self.score(doc, language)

        # if have more than one label with the highest score, return the most frequent label that has the highest score
        '''
        max_lst = []
        for label in pred_score:
            if pred_score[label] == max(pred_score.values()):
                max_lst.append(label)
        if len(max_lst) == 1:
            return max_lst[0]
        elif len(max_lst) > 1:
            d = Counter(train_labels)
            d = sorted(d.items(), key=lambda e:e[1], reverse=True)
            for comb in d:
                for l in max_lst:
                    if comb[0] == l:
                        return l
        '''

        # if have more than one label with the highest score, return a random label that has the highest score
        '''
        return random.choice(max_lst)
        '''

        return max(pred_score, key=pred_score.get)

        # problem 1 the majority class baseline accuracy, return the most frequent language label
            # return "ZHO"


    def test_eval(self, test_docs, test_labels):
        pred_labels = [self.predict(d) for d in test_docs]
        ev = Eval(test_labels, pred_labels)
        return ev.accuracy()


if __name__ == "__main__":
    args = sys.argv[1:]
    niters = int(args[0])


    train_docs, train_labels = load_featurized_docs('train')
    print(len(train_docs), 'training docs with',
        sum(len(d) for d in train_docs)/len(train_docs), 'percepts on avg', file=sys.stderr)

    dev_docs,  dev_labels  = load_featurized_docs('dev')
    print(len(dev_docs), 'dev docs with',
        sum(len(d) for d in dev_docs)/len(dev_docs), 'percepts on avg', file=sys.stderr)


    test_docs,  test_labels  = load_featurized_docs('test')
    print(len(test_docs), 'test docs with',
        sum(len(d) for d in test_docs)/len(test_docs), 'percepts on avg', file=sys.stderr)

    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels)
    test_acc = ptron.test_eval(test_docs, test_labels)
    print("testAcc=", test_acc, file=sys.stderr)
    p4 = ptron.error_analysis(test_docs, test_labels)
    print(p4)

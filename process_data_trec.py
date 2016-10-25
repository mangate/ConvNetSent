__author__ = 'mangate'
import numpy as np
import cPickle
from collections import defaultdict
import re
import pandas as pd
import os.path

from process_data_common import load_bin_vec, add_unknown_words, get_W

classes = {
    'NUM': 0,
    'LOC': 1,
    'HUM': 2,
    'DESC': 3,
    'ENTY': 4,
    'ABBR': 5
    }


def load_train(filename='data/trec/train_5500.label'):
    lines = []
    x = []
    y = []
    with open(filename, 'r') as f:
        lines.extend(f.readlines())
    for line in lines:
        x.append(':'.join(line.split(':')[1:]))
        y.append(classes[line.split(':')[0]])
    return x, y


def load_test(filename='data/trec/TREC_10.label'):
    lines = []
    x = []
    y = []
    with open(filename, 'r') as f:
        lines.extend(f.readlines())
    for line in lines:
        x.append(':'.join(line.split(':')[1:]))
        y.append(classes[line.split(':')[0]])
    return x, y


def build_data_cv(clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    train_x, train_y = load_train()
    test_x, test_y = load_test()
    for i in range(len(train_x)):
        rev = []
        rev.append(train_x[i].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {"y":train_y[i],
                  "text": orig_rev,
                  "num_words": len(orig_rev.split()),
                  "split": 1
                }
        revs.append(datum)
    for i in range(len(test_x)):
        rev = []
        rev.append(test_x[i].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {"y":test_y[i],
                  "text": orig_rev,
                  "num_words": len(orig_rev.split()),
                  "split": 0
                }
        revs.append(datum)
    return revs, vocab


def clean_str(string, TREC=False):
    re.compile('(http|https):\/\/[^\/"]+[^ |"]*')
    string = re.sub('(http|https):\/\/[^\/"]+[^ |"]*', "http", string)
    match = re.search("(#\w+)", string)
    if match is not None:
        for i in match.groups():
            string += " " + i + " "
    string += (" " + "chr200" + " ") * (len([chr(ord(input_x)) for input_x in string if ord(input_x) > 130]) / 4)
    string = ''.join([chr(ord(input_x)) for input_x in string if ord(input_x) < 130]). \
                replace("\\\\", "").replace("&gt;", ">"). \
                replace("&lt;", "<").replace("&amp;", "&")
    string += (" " + "chr201" + " ") * (len(string.split()))
    string += (" " + "chr202" + " ") * string.count(":)")
    string += (" " + "chr203" + " ") * string.count("@")
    string += (" " + "chr204" + " ") * string.count("http")
    string += (" " + "chr205" + " ") * string.count(":(")
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def process_data(file_name):
    if os.path.isfile(file_name):
        print "file {} already exists".format(file_name)
        return

    print "creating dataset..."

    # load data
    print "loading data...",
    revs, vocab = build_data_cv(clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)

    # load word2vec
    print "loading word2vec vectors...",
    w2v_file = 'data/GoogleNews-vectors-negative300.bin'
    w2v = load_bin_vec(w2v_file, vocab)
    print "num words already in word2vec: " + str(len(w2v))
    print "word2vec loaded!"

    #Addind random vectors for all unknown words
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)

    # dump to pickle file
    cPickle.dump([revs, W, W2, word_idx_map, vocab, max_l], open(file_name, "wb"))

    print "dataset created!"


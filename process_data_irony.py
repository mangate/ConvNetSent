__author__ = 'mangate'
import numpy as np
import cPickle
from collections import defaultdict
import re
import pandas as pd
import os.path

from process_data_common import load_bin_vec, add_unknown_words, get_W


def load_dataset(filename='data/irony/irony-labeled.csv'):
    train = pd.read_csv(filename, header=0)
    pos = train[train['label']== 1]
    neg = train[train['label']== -1].sample(len(pos))
    x_pos, y_pos = pos['comment_text'], pos['label']
    x_neg, y_neg = neg['comment_text'], neg['label']

    return list(x_pos), list(x_neg)


def build_data_cv(cv=10,clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    x_pos, x_neg = load_dataset()
    for i in range(len(x_pos)):
        rev = []
        rev.append(x_pos[i].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {"y":1,
                  "text": orig_rev,
                  "num_words": len(orig_rev.split()),
                  "split": np.random.randint(0,cv)
                }
        revs.append(datum)
    for i in range(len(x_neg)):
            rev = []
            rev.append(x_neg[i].strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 0,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)
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
    revs, vocab = build_data_cv(cv=10, clean_string=True)
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


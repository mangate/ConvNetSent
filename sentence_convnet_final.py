__author__ = 'mangate'

import cPickle
from model import Model
import process_data_mr
import process_data_tweets
import process_data_sst1
import process_data_sst2
import process_data_subj
import process_data_trec
import process_data_politeness2
import process_data_opi
import process_data_irony
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

#Flags
tf.flags.DEFINE_boolean("random",False,"Initialize with random word embeddings (default: False)")
tf.flags.DEFINE_boolean("static",False,"Keep the word embeddings static (default: False)")
FLAGS =tf.flags.FLAGS

def evaluate(x, num_classes = 2, k_fold = 10):
    revs, embedding, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    if FLAGS.random:
        embedding = W2
    embedding_dim = 300
    vocab_size = len(vocab) + 1
    filter_sizes = [3, 4, 5]
    num_filters = 100
    vector_length = max_l + 2 * 4
    cnn_model = Model()
    trainable = not FLAGS.static
    cnn_model.build_model(embedding_dim, vocab_size, filter_sizes, num_filters, vector_length, num_classes, trainable)
    cnn_model.run(revs, embedding, word_idx_map, max_l, k_fold)


def evaluate_mr():
    process_data_mr.process_data("data/processed/mr.p")
    x = cPickle.load(open("data/processed/mr.p", "rb"))
    evaluate(x, 2, 10)


def evaluate_tweets():
    process_data_tweets.process_data("data/processed/twitter.p")
    x = cPickle.load(open("data/processed/twitter.p", "rb"))
    evaluate(x, 10, 1)


def evaluate_sst1():
    process_data_sst1.process_data("data/processed/sst1.p")
    x = cPickle.load(open("data/processed/sst1.p", "rb"))
    evaluate(x, 5, 1)


def evaluate_sst2():
    process_data_sst2.process_data("data/processed/sst2.p")
    x = cPickle.load(open("data/processed/sst2.p", "rb"))
    evaluate(x, 2, 1)


def evaluate_subj():
    process_data_subj.process_data("data/processed/subj.p")
    x = cPickle.load(open("data/processed/subj.p", "rb"))
    evaluate(x, 2, 10)


def evaluate_trec():
    process_data_trec.process_data("data/processed/trec.p")
    x = cPickle.load(open("data/processed/trec.p", "rb"))
    evaluate(x, 6, 1)


def evaluate_cr():
    # couldn't find the dataset...
    pass


def evaluate_mpqa():
    # too complicated..
    pass


def evaluate_politeness():
    process_data_politeness2.process_data("data/processed/politeness.p")
    x = cPickle.load(open("data/processed/politeness.p", "rb"))
    evaluate(x, 2, 10)

def evaluate_opi():
    process_data_opi.process_data("data/processed/opi.p")
    x = cPickle.load(open("data/processed/opi.p", "rb"))
    evaluate(x, 6, 10)

def evaluate_irony():
    process_data_irony.process_data("data/processed/irony.p")
    x = cPickle.load(open("data/processed/irony.p", "rb"))
    evaluate(x, 2, 10)

if __name__=="__main__":
   #evaluate_mr()
   #evaluate_tweets()
   #evaluate_sst1()
   #evaluate_sst2()
   #evaluate_subj()
   #evaluate_trec()
   #evaluate_politeness()
   evaluate_irony()





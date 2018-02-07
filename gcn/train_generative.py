from __future__ import division
from __future__ import print_function
import sys
import os

import time
import tensorflow as tf
import scipy.stats as stats

from utils import *
from models import *
from layers import *
from format import *
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 16, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 16, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden6', 16, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden7', 16, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden8', 16, 'Number of units in hidden layer 5.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 5e-12, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('spectral_cap', 9, 'Number of smallest non-zero eigenvalues from each vertex deleted graph')
flags.DEFINE_float('validation', 0.0, 'Percent of training data to withhold for validation')
flags.DEFINE_string('dataset', "mutag", 'Name of dataset to load')
flags.DEFINE_integer('gpu', -1, 'gpu to use, -1 for no gpu')
flags.DEFINE_float('autoregressive_scalar', 0., 'you know')
flags.DEFINE_float('density_scalar', 1., 'you know')
flags.DEFINE_integer('seed', 123, 'TF and numpy seed')

flags.DEFINE_integer('gen_count', 100, 'Number of generated toy graphs for accuracy')
flags.DEFINE_integer('verbose', 1, 'Print shit')
flags.DEFINE_integer('test_count', 1, 'as')
flags.DEFINE_integer('VAE', 1, 'd')

flags.DEFINE_float('p', 0.2, 'p')
flags.DEFINE_integer('d', 4, 'd')

tf.set_random_seed(FLAGS.seed)

densities = np.zeros(FLAGS.test_count)
losses = np.zeros(FLAGS.test_count)

for test_iter in range(FLAGS.test_count):
    FLAGS.seed += 10
    A_orig, A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset, FLAGS.spectral_cap, FLAGS.seed)
    A_orig_, A_, X_, y_train_, _, _, train_mask_, _, _ = load_data(FLAGS.dataset, FLAGS.spectral_cap, FLAGS.seed + 1)

    vertex_count = A.shape[2]
    feature_count = X.shape[2]
    model_func = GraphiteGenModel

    placeholders = {
        'labels': tf.placeholder(tf.float32, shape = (None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'adj_norm': tf.placeholder(tf.float32, shape = (None, vertex_count, vertex_count)),
        'adj_orig': tf.placeholder(tf.float32, shape = (None, vertex_count, vertex_count)),
        'features': tf.placeholder(tf.float32, shape=(None, vertex_count, feature_count)),
        'dropout': tf.placeholder_with_default(0., shape=(), name = "drop"),
        'num_features_nonzero': tf.placeholder(tf.int32, name = "help")
    }

    model = model_func(placeholders, feature_count, vertex_count, logging=True)

    def make_session():
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        if FLAGS.gpu == -1:
            sess = tf.Session()
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu) # Or whichever device you would like to use
            gpu_options = tf.GPUOptions(allow_growth=True)
            sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        return sess

    def evaluate(X, A_norm, A_orig, labels, labels_mask, placeholders, training):
        feed_dict = construct_feed_dict(X, A_norm, A_orig, labels, labels_mask, placeholders)
        if training:
            func = model.opt_op
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        else:
            func = model.loss
        outs_val = sess.run([func, model.loss, model.accuracy, model.log_lik], feed_dict=feed_dict)
        return outs_val[1], outs_val[2], outs_val[3]


    sess = make_session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(FLAGS.epochs):

        val_loss, val_acc, val_log_lik = evaluate(X_, A_, A_orig_, y_train_, train_mask_, placeholders, False)
        train_loss, train_acc, _ = evaluate(X, A, A_orig, y_train, train_mask, placeholders, True)

        if FLAGS.verbose and (epoch + 1) % 50 == 0:
            print("Epoch:", '%04d' % (epoch + 1),"train_loss=", "{:.5f}".format(train_loss),
                                            "val_loss=", "{:.5f}".format(val_loss))


    def plot_graph(A):
        G = nx.from_numpy_matrix(A)
        nx.draw(G)
        plt.show()
        plt.close()


    feed_dict = construct_feed_dict(X_, A_, A_orig_, y_train_, train_mask_, placeholders)
    #gens = sess.run(model.sample_fair(FLAGS.gen_count), feed_dict=feed_dict)
    embs = sess.run(model.z_mean, feed_dict=feed_dict)
    embs = np.mean(embs, 1)

    if FLAGS.verbose:
        print(embs)
        print(y_train_)


    losses[test_iter] = val_log_lik

    # for i in range(5):
    #     A = gens[i]
    #     plot_graph(A)

# print("density")
# print(np.mean(densities))
# print(stats.sem(densities))
print("log_lik")
print(np.mean(losses))
print(stats.sem(losses))



from __future__ import division
from __future__ import print_function
import sys
import os

import time
import tensorflow as tf

import matplotlib.pyplot as plt

from gcn.utils import *
from gcn.models import *

from utils import *
from models import *
from layers import *
from format import *

# Set random seed
seed = 126
#np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 1200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 6, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 4, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0.000, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('learning_rate', 0.007, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 5e-12, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('validation', 0.2, 'Percent of training data to withhold for validation')
flags.DEFINE_string('dataset', "ego", 'Name of dataset to load')
flags.DEFINE_integer('max_dim', 6, 'Maximum vertex count of graph')
flags.DEFINE_integer('training_size', 500, 'Number of training examples')
flags.DEFINE_boolean('plot', False, 'Whether to plot generated graphs')
flags.DEFINE_boolean('save', False, 'Whether to save the plots of generated graphs')
flags.DEFINE_integer('gpu', -1, 'gpu to use, -1 for no gpu')

if FLAGS.dataset == "mutag":
    read_func = read_mutag
elif FLAGS.dataset == "clintox":
    read_func = read_clintox
elif FLAGS.dataset == "ego":
    read_func = read_ego
elif FLAGS.dataset == "star":
    read_func = read_star
elif FLAGS.dataset == "ring":
    read_func = read_ring
elif FLAGS.dataset == "bipartite":
    read_func = read_bipartite

# Load data
labels, A_norm, X, train_mask, val_mask, vertex_count, feature_count = load_generative_data(read_func)

model_func = GenerativeGCN

# Define placeholders

placeholders = {
    'labels': tf.placeholder(tf.float32, shape = None),
    'adj_norm': tf.placeholder(tf.float32, shape = (None, vertex_count, vertex_count)),
    'features': tf.placeholder(tf.float32, shape=(None, vertex_count, feature_count)),
    'dropout': tf.placeholder_with_default(0., shape=(), name = "drop"),
    'num_features_nonzero': tf.placeholder(tf.int32, name = "help")  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=feature_count, vertex_count = vertex_count, logging=True)


if FLAGS.gpu == -1:
    sess = tf.Session()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) # Or whichever device you would like to use
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))




# Define model evaluation function
def evaluate(X, labels, A_norm, placeholders, training):
    feed_dict = construct_generative_feed_dict(X, labels, A_norm, placeholders)
    if training:
        func = model.opt_op
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    else:
        func = model.data
    outs_val = sess.run([func, model.loss], feed_dict=feed_dict)
    return outs_val[1]

# Init variables
sess.run(tf.global_variables_initializer())

cost_train = []
cost_val = []

indices_t = np.where(train_mask)[0]
indices_v = np.where(val_mask)[0]
X_t, labels_t, A_norm_t = X[indices_t], labels[indices_t], A_norm[indices_t]
X_v, labels_v, A_norm_v = X[indices_v], labels[indices_v], A_norm[indices_v]

# Train model
for epoch in range(FLAGS.epochs):
    cost = evaluate(X_t, labels_t, A_norm_t, placeholders, False)
    cost_val.append(cost)
    outs = evaluate(X_v, labels_v, A_norm_v, placeholders, True)
    cost_train.append(outs)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs), "val_loss=", "{:.5f}".format(cost))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def is_accurate(G):
    if FLAGS.dataset == "ego":
        A = nx.to_numpy_matrix(G)
        np.fill_diagonal(A, 1)
        return np.max(np.min(A, 0))
    elif FLAGS.dataset == "star":
        H = nx.star_graph(G.order() - 1)
        return nx.is_isomorphic(G,H)
    elif FLAGS.dataset == "ring":
        cycles = nx.cycle_basis(G)
        return len(cycles) == 1 and len(cycles[0]) == G.order()

def generate():
    partial = np.zeros((vertex_count, vertex_count))
    partial_feature = np.zeros((1, vertex_count, feature_count))
    partial_norm = np.zeros((1, vertex_count, vertex_count))
    for r in range(vertex_count):
        for c in range(r):
            partial_norm[0] = preprocess_adj(partial).todense()
            helper_features = make_helper_features(vertex_count, r, c)
            partial_feature[0] = np.hstack((np.identity(vertex_count), helper_features))

            feed_dict = construct_generative_feed_dict(partial_feature, -1, partial_norm, placeholders)
            pred = sess.run([model.pred], feed_dict=feed_dict)
            pred = sigmoid(pred[0])
            label = np.random.choice(2, p = [1-pred, pred])
            partial[r,c] = partial[c,r] = label
            # endbit = sigmoid(endbit)
            # end = np.random.choice(2, p = [1-endbit, endbit])
            # if end == 1:
            #     break
    G = nx.from_numpy_matrix(partial)
    return G

graphs = []
for i in range(100):
    graphs.append(generate())

acc = [is_accurate(G) for G in graphs]
print(np.mean(np.array(acc)))

if not FLAGS.plot:
    sys.exit()

for i in range(len(graphs)):
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.subplot(4, 5, i + 1)
    G = graphs[i]
    nx.draw_networkx(G, pos=nx.spring_layout(G), node_size = 40, font_size=7)

plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
title = FLAGS.dataset + str(vertex_count)
plt.suptitle(title)
if FLAGS.save:
    plt.savefig("saved/" + title)
plt.show()
plt.close()

plt.plot(cost_train)
plt.plot(cost_val)
plt.legend(['train', 'validation'], loc='upper left')
if FLAGS.save:
    plt.savefig("saved/" + title + " graph")
plt.show()
plt.close()


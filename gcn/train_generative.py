from __future__ import division
from __future__ import print_function
import sys
import os

import time
import tensorflow as tf


from gcn.utils import *
from gcn.models import *

from utils import *
from models import *
from layers import *
from format import *

# Set random seed
#seed = 126
#np.random.seed(seed)
#tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 7, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 7, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 7, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0.001, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 5e-12, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('validation', 0.2, 'Percent of training data to withhold for validation')
flags.DEFINE_integer('test', 3, "Number of molecules to holdout for full generation testing")
flags.DEFINE_string('dataset', "ego", 'Name of dataset to load')
flags.DEFINE_integer('max_dim', 12, 'Maximum vertex count of graph')
flags.DEFINE_integer('training_size', 300, 'Number of training examples')
flags.DEFINE_boolean('plot', False, 'Whether to plot generated graphs')
flags.DEFINE_boolean('save', False, 'Whether to save the plots of generated graphs')
flags.DEFINE_integer('gpu', -1, 'gpu to use, -1 for no gpu')
flags.DEFINE_integer('batch_size', 30, 'size of each batch to gradient descent')

import matplotlib
if FLAGS.gpu != -1:
    matplotlib.use('agg')
import matplotlib.pyplot as plt

if FLAGS.dataset == "mutag":
    read_func = read_mutag
elif FLAGS.dataset == "clintox":
    read_func = read_zinc
else:
    read_func = read_toy

# Load data
labels, A_norm, X, train_mask, val_mask, vertex_count, feature_count, A_test, X_test = load_generative_data(read_func)

model_func = GenerativeGCN

# Define placeholders

placeholders = {
    'labels': tf.placeholder(tf.float32, shape = (None, labels.shape[1])),
    'adj_norm': tf.placeholder(tf.float32, shape = (None, vertex_count, vertex_count)),
    'features': tf.placeholder(tf.float32, shape=(None, vertex_count, feature_count)),
    'dropout': tf.placeholder_with_default(0., shape=(), name = "drop"),
    'num_features_nonzero': tf.placeholder(tf.int32, name = "help")  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=feature_count, vertex_count = vertex_count, logging=True)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
if FLAGS.gpu == -1:
    sess = tf.Session()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu) # Or whichever device you would like to use
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
    offset = (epoch * FLAGS.batch_size) % (labels_t.shape[0] - FLAGS.batch_size)
    indices = range(offset,(offset + FLAGS.batch_size))

    cost = evaluate(X_t[indices], labels_t[indices], A_norm_t[indices], placeholders, False)
    cost_val.append(cost)
    outs = evaluate(X_v, labels_v, A_norm_v, placeholders, True)
    cost_train.append(outs)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs), "val_loss=", "{:.5f}".format(cost))

    if FLAGS.early_stopping != -1 and epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def is_accurate(G):
    if G.order() == 0:
        return 0

    if FLAGS.dataset == "ego":
        A = nx.to_numpy_matrix(G)
        np.fill_diagonal(A, 1)
        return np.max(np.min(A, 0))
    elif FLAGS.dataset == "star":
        H = nx.star_graph(G.order() - 1)
        return nx.is_isomorphic(G,H)
    elif FLAGS.dataset == "ring":
        H = nx.cycle_graph(G.order())
        return nx.is_isomorphic(G,H)
    elif FLAGS.dataset == "lollipop":
        if G.order() < 3:
            return 0
        H = nx.lollipop_graph(3, G.order() - 3)
        return nx.is_isomorphic(G,H)
    else:
        return 0

def link_acc(As, Xs):
    pos = []
    neg = []
    for A, X in zip(As, Xs):
        for r in range(A.shape[0]):
            for c in range(r):
                partial = np.copy(A)
                partial[r,c] = 0
                partial_norm = np.zeros((1, vertex_count, vertex_count))
                partial_norm[0] = preprocess_adj(partial).todense()
                dummy_label = np.zeros((1,2))
                hit_nodes = np.zeros(vertex_count) + 1
                hit_nodes[r] = hit_nodes[c] = 0
                helper_features = make_helper_features(vertex_count, r, c, hit_nodes)
                partial_feature = np.zeros((1, vertex_count, feature_count))
                partial_feature[0] = np.hstack((X, np.identity(vertex_count), helper_features))

                feed_dict = construct_generative_feed_dict(partial_feature, dummy_label, partial_norm, placeholders)
                pred = sess.run([model.pred], feed_dict=feed_dict)
                pred = softmax(pred[0])
                label = np.random.choice(len(pred), p = pred)

                point = 1.0 if label == A[r,c] else 0.0
                if A[r,c] == 1:
                    pos.append(point)
                else:
                    neg.append(point)
    return (np.mean(np.array(pos)), np.mean(np.array(neg)))

def toy_generate():
    partial = np.zeros((vertex_count, vertex_count))
    partial_feature = np.zeros((1, vertex_count, feature_count))
    partial_norm = np.zeros((1, vertex_count, vertex_count))
    dummy_label = np.zeros((1, 2))

    hit_nodes = np.zeros(vertex_count)
    q = Queue.Queue()
    enqueued = set()
    q.put(0)
    enqueued.add(0)

    while not q.empty():
        r = q.get()
        for c in range(vertex_count):
            if hit_nodes[c] == 1 or c == r:
                continue

            partial_norm[0] = preprocess_adj(partial).todense()
            helper_features = make_helper_features(vertex_count, r, c, hit_nodes)
            partial_feature[0] = np.hstack((np.identity(vertex_count), helper_features))
            feed_dict = construct_generative_feed_dict(partial_feature, dummy_label, partial_norm, placeholders)
            pred = sess.run([model.pred], feed_dict=feed_dict)
            pred = softmax(pred[0])
            label = np.random.choice(len(pred), p = pred)

            if label != 0:
                partial[r,c] = partial[c,r] = 1
            if label != 0 and c not in enqueued:
                q.put(c)
                enqueued.add(c)
        hit_nodes[r] = 1

    G = nx.from_numpy_matrix(partial)
    G.remove_nodes_from(nx.isolates(G))
    return G

def generate(X):
    partial = np.zeros((vertex_count, vertex_count))
    partial_feature = np.zeros((1, vertex_count, feature_count))
    partial_norm = np.zeros((1, vertex_count, vertex_count))
    dummy_label = np.zeros((1, 2))
    bond_dic = {}

    hit_nodes = np.zeros(vertex_count)
    q = Queue.Queue()
    enqueued = set()
    q.put(0)
    enqueued.add(0)

    while not q.empty():
        r = q.get()
        for c in range(vertex_count):
            if hit_nodes[c] == 1 or c == r:
                continue

            partial_norm[0] = preprocess_adj(partial).todense()
            helper_features = make_helper_features(vertex_count, r, c, hit_nodes)
            partial_feature[0] = np.hstack((X, np.identity(vertex_count), helper_features))
            feed_dict = construct_generative_feed_dict(partial_feature, dummy_label, partial_norm, placeholders)
            pred = sess.run([model.pred], feed_dict=feed_dict)
            pred = softmax(pred[0])
            label = np.random.choice(len(pred), p = pred)

            if label != 0:
                bond_dic[(r,c)] = bond_dic[(c,r)] = label
                partial[r,c] = partial[c,r] = 1
            if label != 0 and c not in enqueued:
                q.put(c)
                enqueued.add(c)
        hit_nodes[r] = 1

    G = build_molecule_graph(partial, X)
    return G

if FLAGS.dataset != "clintox":
    acc = [is_accurate(toy_generate()) for i in range(100)]
    print (np.mean(np.array(acc)))
else:
    print(link_acc(A_test[:FLAGS.test], X_test[:FLAGS.test]))
    print(link_acc(A_test[FLAGS.test:], X_test[FLAGS.test:]))

if not FLAGS.plot and not FLAGS.save:
    sys.exit()

def plot_graph(G, i):
    G.remove_nodes_from(nx.isolates(G))
    plt.subplot(2, 3, i)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    labels = nx.get_node_attributes(G, "atom")
    nx.draw_networkx(G, labels = labels, node_size = 50, font_size=7)

for i in range(2 * FLAGS.test):
    A = A_test[i]
    X = X_test[i]
    real = build_molecule_graph(A, X)
    plot_graph(real, 1)
    for j in range(2, 7):
        pred = generate(X)
        plot_graph(pred, j)

    title = FLAGS.dataset + str(i)
    title += "test" if i < FLAGS.test else "train"
    plt.suptitle(title)
    if FLAGS.save:
        plt.savefig("saved/" + title)
    if FLAGS.plot:
        plt.show()
    plt.close()

plt.plot(cost_val)
plt.plot(cost_train)
plt.legend(['validation', 'train'], loc='upper left')
if FLAGS.save:
    plt.savefig("saved/" + title + "_graph")
if FLAGS.plot:
    plt.show()
plt.close()


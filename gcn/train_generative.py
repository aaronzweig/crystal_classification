from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import *

from utils import *
from models import *
from layers import *
from format import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 900, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 54, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.00, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 5e-12, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('validation', 0.2, 'Percent of training data to withhold for validation')
flags.DEFINE_string('dataset', "ego", 'Name of dataset to load')

if FLAGS.dataset == "mutag":
    read_func = read_mutag
elif FLAGS.dataset == "clintox":
    read_func = read_clintox
elif FLAGS.dataset == "ego":
    read_func = read_ego

# Load data
A, A_norm, X, train_mask, val_mask, vertex_count, feature_count = load_generative_data(read_func)

model_func = GenerativeGCN

# Define placeholders

placeholders = {
    'adj': tf.placeholder(tf.float32, shape = (vertex_count, vertex_count)),
    'adj_norm': tf.placeholder(tf.float32, shape = (vertex_count-1, vertex_count, vertex_count)),
    'features': tf.placeholder(tf.float32, shape=(vertex_count-1, vertex_count, feature_count)),
    'dropout': tf.placeholder_with_default(0., shape=(), name = "drop"),
    'num_features_nonzero': tf.placeholder(tf.int32, name = "help")  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=feature_count, vertex_count = vertex_count, logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(X, A, A_norm, mask, placeholders):
    outs = []
    accuracies = []
    for i in np.where(mask)[0]:
        feed_dict_val = construct_generative_feed_dict(X[i], A[i], A_norm[i], placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        outs.append(outs_val[0])
        accuracies.append(outs_val[1])
    return np.mean(np.array(outs)), np.mean(np.array(accuracies))

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    outs_list = []

    for i in np.where(train_mask)[0]:
        feed_dict = construct_generative_feed_dict(X[i], A[i], A_norm[i], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
        outs_list.append(outs[1])
    outs = np.mean(np.array(outs_list))

    cost, accuracy = evaluate(X, A, A_norm, val_mask, placeholders)
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs), "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(accuracy))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for j in np.where(val_mask)[0][:12]:
    A_partial = np.zeros((vertex_count, vertex_count))
    A_partial_norm = np.zeros((vertex_count-1, vertex_count, vertex_count))
    #A_partial_norm = A_norm[j]
    for i in range(1,vertex_count):
        A_partial_norm[i-1,:i,:i] = preprocess_adj(A_partial[:i,:i]).todense()
        feed_dict = construct_generative_feed_dict(X[j], A_partial, A_partial_norm, placeholders)
        outs = sess.run([model.outputs], feed_dict=feed_dict)
        labels = sigmoid(outs[0][i-1].flatten())
        labels = [np.random.choice(2, p = [1-k, k]) for k in labels]
        A_partial[i,:i] = labels[:i]
        A_partial[:i,i] = labels[:i]
    print("compare")
    #print(A[j])
    print(A_partial)

# # Testing
# test_cost, test_acc, test_duration = evaluate(X, A, A_norm, test_mask, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

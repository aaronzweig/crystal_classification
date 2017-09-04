from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import *

from utils import *
from models import *
from layers import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 5e-12, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_vertices', 30, 'Maximum size of possible graphs')
flags.DEFINE_integer('additional_features', 1, 'Features beyond just the identity matrix')

# Load data
A, X, labels, train_mask, val_mask, test_mask = load_global_data(FLAGS.max_vertices)

model_func = GlobalGCN

# Define placeholders
MAX = FLAGS.max_vertices
ADD = FLAGS.additional_features
placeholders = {
    'support': [tf.placeholder(tf.float32, shape = (None, MAX, MAX))],
    'features': tf.placeholder(tf.float32, shape=(None, MAX, MAX+ADD)),
    'labels': tf.placeholder(tf.float32, shape = (None, 2)),
    'labels_mask': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=FLAGS.max_vertices + FLAGS.additional_features, logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(X, A, labels, labels_mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(X, [A], labels, labels_mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary

    feed_dict = construct_feed_dict(X, [A], labels, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(X, A, labels, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(X, A, labels, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

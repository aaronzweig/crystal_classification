import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import Queue

import format

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS




def load_global_data(read_func):
    As, Xs, labels = read_func()
    As = [np.asarray(preprocess_adj(A).todense()) for A in As]
    Xs = [np.asarray(X) for X in Xs]

    A = np.dstack(tuple(As))
    A = np.transpose(A, axes=(2, 0, 1))
    X = np.dstack(tuple(Xs))
    X = np.transpose(X, axes=(2, 0, 1))

    count = labels.shape[0]
    train_mask = np.random.choice(2, count, p=[FLAGS.validation, 1 - FLAGS.validation])
    val_mask = 1 - train_mask
    train_mask = np.array(train_mask, dtype=np.bool)
    #TODO: Have separate testing data for final evaluation
    val_mask = test_mask = np.array(val_mask, dtype=np.bool)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask, A.shape[2], X.shape[2]

def make_helper_features(dim, r, c, hit_nodes):
    features = np.zeros((dim, 4))
    features[r,0] = 1
    features[c,1] = 1
    features[:c,2] = 1
    features[:,3] = hit_nodes
    return features

def load_generative_data(read_func):
    As, Xs, dist = read_func()
    dim = FLAGS.max_dim
    batch = len(As)
    feature_count = Xs[0].shape[1]

    labels = []
    A_norm = []
    X = []

    for i in range(batch):
        if i % (batch / 10) == 0:
            print(str(i) + "/" + str(batch))
        temp = As[i]
        adj = np.zeros((dim, dim))
        adj[:temp.shape[0], :temp.shape[1]] = temp

        features = Xs[i]
        partial = np.zeros((dim, dim))
        hit_nodes = np.zeros(dim)

        q = Queue.Queue()
        enqueued = set()
        q.put(0)
        enqueued.add(0)

        while not q.empty():
            r = q.get()
            for c in range(dim):
                if hit_nodes[c] == 1 or c == r:
                    continue

                if FLAGS.dataset == "clintox":
                    label = np.zeros(5)
                else:
                    label = np.zeros(2)


                adj_norm = np.asarray(preprocess_adj(partial).todense())
                label[int(adj[r,c])] = 1
                helper_features = make_helper_features(dim, r, c, hit_nodes)
                updated_feature = np.asarray(np.hstack((features, helper_features)))

                perm = np.random.permutation(dim)
                adj_norm = adj_norm[:,perm]
                adj_norm = adj_norm[perm, :]
                updated_feature = updated_feature[perm, :]

                X.append(updated_feature)
                A_norm.append(adj_norm)
                labels.append(label)

                partial[r,c] = partial[c,r] = 0 if int(adj[r,c]) == 0 else 1
                if label[0] == 0 and c not in enqueued:
                    q.put(c)
                    enqueued.add(c)
            hit_nodes[r] = 1

    labels = np.vstack(tuple(labels))
    A_norm = np.dstack(tuple(A_norm))
    A_norm = np.transpose(A_norm, axes=(2, 0, 1))
    X = np.dstack(tuple(X))
    X = np.transpose(X, axes=(2, 0, 1))

    train_mask = np.random.choice(2, X.shape[0], p=[FLAGS.validation, 1 - FLAGS.validation])
    val_mask = 1 - train_mask
    train_mask = np.array(train_mask, dtype=np.bool)
    val_mask = np.array(val_mask, dtype=np.bool)

    return labels, A_norm, X, train_mask, val_mask, A_norm.shape[2], X.shape[2], dist


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def construct_generative_feed_dict(features, labels, adj_norm, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['adj_norm']: adj_norm})
    feed_dict.update({placeholders['num_features_nonzero']: np.asarray(features.shape)})
    return feed_dict


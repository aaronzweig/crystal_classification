import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

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

def load_generative_data(read_func):
    As, Xs = read_func()
    batch = len(As)
    dim = As[0].shape[0]
    feature_count = Xs[0].shape[1]

    A = np.zeros((batch, dim, dim))
    A_norm = np.zeros((batch, dim-1, dim, dim))
    X = np.zeros((batch, dim-1, dim, feature_count))

    for i in range(batch):
        adj = As[i]
        feature = Xs[i]
        A[i,:,:] = adj
        for j in range(1, dim):
            adj_norm = preprocess_adj(adj[:j,:j]).todense()
            A_norm[i,j-1,:j,:j] = adj_norm
            X[i,j-1,:,:] = feature

    train_mask = np.random.choice(2, batch, p=[FLAGS.validation, 1 - FLAGS.validation])
    val_mask = 1 - train_mask
    train_mask = np.array(train_mask, dtype=np.bool)
    val_mask = np.array(val_mask, dtype=np.bool)

    return A, A_norm, X, train_mask, val_mask, dim, feature_count

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

def construct_generative_feed_dict(features, adj, adj_norm, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['adj_norm']: adj_norm})
    feed_dict.update({placeholders['num_features_nonzero']: np.asarray(features.shape)})
    return feed_dict


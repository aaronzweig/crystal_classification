import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import Queue
import pickle

from format import *

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS




def load_data(dataset, spectral_cap):
    As, Xs, labels = read_dataset(dataset, spectral_cap)
    A_origs = [np.asarray(A) for A in As]
    As = [np.asarray(preprocess_adj(A).todense()) for A in As]
    Xs = [np.asarray(X) for X in Xs]

    A_orig = np.dstack(tuple(A_origs))
    A_orig = np.transpose(A_orig, axes=(2, 0, 1))
    A = np.dstack(tuple(As))
    A = np.transpose(A, axes=(2, 0, 1))
    X = np.dstack(tuple(Xs))
    X = np.transpose(X, axes=(2, 0, 1))

    train_mask = np.random.choice(2, labels.shape[0], p=[FLAGS.validation, 1 - FLAGS.validation])
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

    return A_orig, A, X, y_train, y_val, y_test, train_mask, val_mask, test_mask


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


def construct_feed_dict(features, adj_norm, adj_orig, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj_norm']: adj_norm})
    feed_dict.update({placeholders['adj_orig']: adj_orig})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


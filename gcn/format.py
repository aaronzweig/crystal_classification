import numpy as np
from scipy.sparse import csr_matrix, hstack, identity
from scipy.sparse.csgraph import laplacian
from sklearn import manifold
import csv
import pickle
import networkx as nx

# import graphkernels.kernels as gk
# import igraph

from format import *

# import tensorflow as tf
# flags = tf.app.flags
# FLAGS = flags.FLAGS

dim = 20
lower_dim = 10

def read_labels(labels):

	remap = {}
	for label in labels:
		if label not in remap:
			remap[label] = len(remap)

	new_labels = np.zeros((len(labels), len(remap)))
	for i in range(len(labels)):
		label = labels[i]
		new_labels[i][remap[label]] = 1

	return new_labels

def pad(A, X, size):
	A_prime = np.zeros((size, size))
	X_prime = np.zeros((size, X.shape[1] + 1))

	A_prime[:A.shape[0], :A.shape[1]] = A
	X_prime[:A.shape[0], :-1] = X
	X_prime[A.shape[0]:, -1] += 1

	return A_prime, X_prime

def permute(A, X):
	P = np.identity(A.shape[0])
	np.random.shuffle(P)
	return np.dot(P, A).dot(P.T), np.dot(P, X)

def get_cofactor(A, i):
	indices = range(A.shape[0])
	indices.remove(i)
	cofactor = A[indices, :]
	cofactor = cofactor[:, indices]
	return cofactor

def distance_features(A,k):
	G = nx.from_numpy_matrix(A)
	D = nx.floyd_warshall_numpy(G)
	return D[:,:k]

def spectral_clustering_features(A, k):
	features = manifold.spectral_embedding(csr_matrix(A), n_components=k)
	return features

def spectral_features(A, k):
	n = A.shape[0]

	features = np.zeros((n, k))
	L = laplacian(A, normed = True)
	for i in range(n):
		cofactor = get_cofactor(L, i)
		v, _ = np.linalg.eig(cofactor)
		v = np.sort(v)
		v = np.trim_zeros(v)

		#might be too small to have enough eigenvalues, pad the rest as zero
		length = min(k, len(v))

		features[i,:length] = v.real[:length]
	return features


def read_siemens():
	As = []
	Xs = []

	for i in range(100):
		nodes = []
		prefix = "Successful" if i >= 143 else "Unsuccessful"
		with open(prefix + "/nodes_case" + str(i) + ".txt") as f:
			nodes = f.read().splitlines()
			nodes = [int(n) for n in nodes]
		with open(prefix + "/edges_case" + str(i) + ".txt") as f:
			edges = f.read().splitlines()
			edges = [n.strip('[]').strip().split(',') for n in edges]
			edges = [(int(n[0]), int(n[1])) for n in edges]

		# A = np.zeros((250, 250))
		# for edge in edges:
		# 	v1 = edge[0]
		# 	v2 = edge[1]
		# 	A[v1, v2] = 1
		# X = np.identity(250)
		# feature = 1 if i >= 143 else 0
		# X = np.hstack((X, np.zeros((A.shape[0], 1)) + feature))

		G = nx.cycle_graph(10)
		A = nx.to_numpy_matrix(G)
		X = np.identity(10)

		As.append(A)
		Xs.append(X)

	return As, Xs, np.zeros((len(As), 2))		


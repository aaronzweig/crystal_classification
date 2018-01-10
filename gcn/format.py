import numpy as np
from scipy.sparse import csr_matrix, hstack, identity
from scipy.sparse.csgraph import laplacian
from sklearn import manifold
import csv
import pickle
import networkx as nx

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
	X_prime = np.zeros((size, X.shape[1]))

	A_prime[:A.shape[0], :A.shape[1]] = A
	X_prime[:A.shape[0], :] = X

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

def read_dataset(dataset, spectral_cap):
	f = open("datasets/" + dataset + ".graph")
	data = pickle.loads(f.read())
	f.close()

	labels = read_labels(data['labels'])
	graphs = data['graph']

	pad_size = 0
	for key in graphs.keys():
		graph = graphs[key]
		vertex_count = len(graph.keys())
		pad_size = max(pad_size, vertex_count)

	As = []
	Xs = []

	for key in graphs.keys():
		graph = graphs[key]
		vertex_count = len(graph.keys())
		feature_count = len(graph[0]['label'])

		A = np.zeros((vertex_count, vertex_count))
		X = np.zeros((vertex_count, feature_count))
		for i in range(vertex_count):
			X[i] = graph[i]['label']
			for j in graph[i]['neighbors']:
				A[i,j] = 1

		A, X = permute(A, X)

		s_features = spectral_features(A, spectral_cap)
		X = np.hstack((X, s_features))
	
		A, X = pad(A, X, pad_size)

		As.append(A)
		Xs.append(X)

	return As, Xs, labels

def read_mutag():
	#hard-coded maximum vertex count specifically for the mutag dataset
	PAD_SIZE = 30

	def clean(L):
		return L.split('\n')[1:-1]

	As = []
	Xs = []
	Cs = np.zeros((188, 2))
	for i in range(1,189):
		
		with open("mutag/mutag_" + str(i) + ".graph") as f:
			V, E, C = f.read().split('#')[1:4]
			V = clean(V)
			E = clean(E)
			C = clean(C)

			row, col, data = zip(*[e.split(",") for e in E])

			V = [int(val) for val in V]
			row = [int(val) - 1 for val in row]
			col = [int(val) - 1 for val in col]
			data = [int(val) for val in data]
			C = 1 if C[0] == '1' else 0
			Cs[i-1,C] = 1


			A = csr_matrix((data,(row,col)), dtype = 'int16')
			row = range(len(V))
			col = [0] * len(V)
			X = csr_matrix((V,(row,col)), dtype = 'int16')


			A, X = A.todense(), X.todense()
			A, X = permute(A, X)

			s_features = spectral_features(A, 9)
			X = np.hstack((X, s_features))
			A, X = pad(A, X, PAD_SIZE)

			As.append(A)
			Xs.append(X)
	return As, Xs, np.stack(Cs)

def read_toy():
	dim = FLAGS.max_dim
	batch = FLAGS.training_size

	As = []
	Xs = []

	for i in range(batch):
		local_dim = np.random.randint(3,dim + 1)
		if FLAGS.dataset == "star":
			G = nx.star_graph(local_dim - 1)
			G = reorder_graph(G)
		elif FLAGS.dataset == "ring":
			G = nx.cycle_graph(local_dim)
			G = reorder_graph(G)
		elif FLAGS.dataset == "ego":
			G = nx.fast_gnp_random_graph(local_dim, 0.1)
			H = nx.star_graph(local_dim - 1)
			G = nx.compose(G,H)
			G = reorder_graph(G)
		elif FLAGS.dataset == "lollipop":
			G = nx.lollipop_graph(3, local_dim - 3)
			G = reorder_graph(G)

		A = nx.to_numpy_matrix(G)

		Apad = np.zeros((dim, dim))
		Apad[:A.shape[0], :A.shape[1]] = A
		As.append(Apad)
		Xs.append(0)

	return As, Xs, dim

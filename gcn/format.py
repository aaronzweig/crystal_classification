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

def read_siemens():
	As = []
	Xs = []
	types = np.zeros((2000-143, 2))

	for i in range(143,2000):
		nodes = []
		with open("Successful/nodes_case" + str(i) + ".txt") as f:
			nodes = f.read().splitlines()
			nodes = [int(n) for n in nodes]
		with open("Successful/edges_case" + str(i) + ".txt") as f:
			edges = f.read().splitlines()
			edges = [n.strip('[]').strip().split(',') for n in edges]
			edges = [(int(n[0]), int(n[1])) for n in edges]
		A = np.zeros((200, 200))
		for edge in edges:
			v1 = nodes.index(edge[0])
			v2 = nodes.index(edge[1])
			A[v1, v2] = 1
		X = np.identity(200)

		As.append(A)
		Xs.append(X)

	return As, Xs, types		

read_siemens()

def read_toy(dataset_real, spectral_cap, seed = None):
	batch = 100
	p = FLAGS.p
	d = FLAGS.d
	np.random.seed(seed)

	As = []
	Xs = []
	types = np.zeros((batch, 2))

	dic = {}
	dic['ego'] = 0
	dic['ER'] = 1
	dic['regular'] = 2
	dic['geometric'] = 3
	dic['power_tree'] = 4
	dic['barabasi'] = 5

	for i in range(batch):
		local_dim = np.random.randint(lower_dim ,dim + 1)
		
		dataset = dataset_real
		if dataset == "all":
			random_choice = np.random.randint(4)
			dataset = ['ego', 'ER', 'regular', 'geometric'][random_choice]

		if dataset == "ego":
			G = nx.fast_gnp_random_graph(local_dim, p)
			H = nx.star_graph(local_dim - 1)
			G = nx.compose(G,H)
		elif dataset == "ER":
			G = nx.fast_gnp_random_graph(local_dim, p)
		elif dataset == "regular":
			G = nx.random_regular_graph(d, local_dim)
		elif dataset == "geometric":
			G = nx.random_geometric_graph(local_dim, p)
		elif dataset == "power_tree":
			G = nx.random_powerlaw_tree(local_dim, tries = 100000)
		elif dataset == "barabasi":
			G = nx.barabasi_albert_graph(local_dim, d)		

		A = nx.to_numpy_matrix(G)
		X = np.zeros((A.shape[0], 1))

		A, X = permute(A, X)
		A, X = pad(A, X, dim)
		X = np.hstack((X, np.identity(dim)))

		As.append(A)
		Xs.append(X)
		types[i][0] = dic[dataset]

	return As, Xs, types

# def kernel_scores(generated, test):
# 	adj_list = []
# 	for A in generated:
# 		np.fill_diagonal(A, 0)
# 		A = A.astype(int)
# 		graph = igraph.Graph.Adjacency((A > 0).tolist(), mode = igraph.ADJ_MAX)
# 		graph.vs['id'] = [str(i) for i in range(A.shape[0])]
# 		graph.vs['label'] = [0 for i in range(A.shape[0])]
# 		graph.es['label'] = [1 for i in range(np.sum(A))]
# 		adj_list.append(graph)
# 	for A in test:
# 		np.fill_diagonal(A, 0)
# 		A = A.astype(int)
# 		graph = igraph.Graph.Adjacency((A > 0).tolist(), mode = igraph.ADJ_MAX)
# 		graph.vs['id'] = [str(i) for i in range(A.shape[0])]
# 		graph.vs['label'] = [0 for i in range(A.shape[0])]
# 		graph.es['label'] = [1 for i in range(np.sum(A))]
# 		adj_list.append(graph)

# 	adj_list = np.asarray(adj_list)

# 	funcs = [gk.CalculateWLKernel, gk.CalculateGraphletKernel, gk.CalculateShortestPathKernel]
# 	kernels = []
# 	for func in funcs:
# 		K = func(adj_list)
# 		norms = np.sqrt(np.diagonal(K))
# 		norms = np.outer(norms, norms)
# 		K = K / norms
# 		K = K[:len(generated), len(generated):]
# 		kernels.append(np.mean(K))
# 	return kernels

def density_estimate(gens):
	prob = 0
	for i in range(lower_dim, dim + 1):
		prob += (1.0 / (dim - lower_dim)) * i**2 / dim**2
	return prob * np.mean(gens)

def degree_estimate(gens):
	prob = 0
	for i in range(lower_dim, dim + 1):
		prob += (1.0 / (dim - lower_dim)) * i**2 / dim**2
	return prob * np.mean(np.sum(gens, 1))

def accurate_toy(dataset, G):
    if G.order() == 0:
        return 0

    if dataset == "ego":
        A = nx.to_numpy_matrix(G)
        np.fill_diagonal(A, 0)

        n = A.shape[0]

        alive_nodes = np.count_nonzero(np.sum(A, 0))
        return np.max(np.sum(A, 0)) + 1 == alive_nodes
    elif dataset == "ring":
        H = nx.cycle_graph(G.order())
        return nx.is_isomorphic(G,H)
    else:
        return 0

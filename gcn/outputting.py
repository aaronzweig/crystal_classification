import numpy as np
from scipy.sparse import csr_matrix, hstack, identity
from scipy.sparse.csgraph import laplacian
import csv
import pickle

graphs = np.load("samples.npy")

for i in range(100):
	graph = graphs[i]
	f = open("generated/nodes_case" + str(i) + ".txt", "w")
	for k in range(graph.shape[0]):
		f.write(str(k) + "\n")
	f.close()

	f = open("generated/edges_case" + str(i) + ".txt", "w")
	for r in range(graph.shape[0]):
		for c in range(graph.shape[1]):
			if graph[r][c] == 1:
				f.write("[" + str(r) + ", " + str(c) + "]\n")
	f.close()

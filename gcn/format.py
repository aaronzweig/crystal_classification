from pynauty import Graph, certificate, autgrp
import numpy as np
from scipy.sparse import csr_matrix, hstack, identity

def clean(L):
	return L.split('\n')[1:-1]

def add(a,b,dic):
	dic[a] = dic.get(a, []) + [b]

def read_mutag():
	#hard-coded maximum vertex count specifically for the mutag dataset
	VERTICES = 30

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

			adjacency_dict = {}
			for j in range(len(row)):
				add(row[j],col[j],adjacency_dict)

			g = Graph(VERTICES, False, adjacency_dict)
			labeling = certificate(g)
			row = [labeling.index(val) for val in row]
			col = [labeling.index(val) for val in col]
			A = csr_matrix((data,(row,col)), shape=(VERTICES, VERTICES), dtype = 'int16')

			row = [labeling.index(val) for val in range(len(V))]
			col = [0] * len(V)
			X = csr_matrix((V,(row,col)), shape=(VERTICES, 1), dtype = 'int16')
			X = hstack((X, identity(VERTICES, dtype='int8', format='csr')), format='csr')

			As.append(A.todense())
			Xs.append(X.todense())
	return As, Xs, np.stack(Cs)

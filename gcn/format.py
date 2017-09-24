# from pynauty import Graph, certificate, autgrp
import numpy as np
from scipy.sparse import csr_matrix, hstack, identity
from rdkit import Chem
import csv
import networkx as nx

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

##########################################################################
#Partitally from "Convolutional Networks on Graphs for Learning Molecular Fingerprints"

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)

def atom_features(atom):
	recognized_elements = 	['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
							'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
							'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
							'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
							'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
	features = [atom.GetDegree(),
    			atom.GetTotalNumHs(),
    			atom.GetImplicitValence(),
    			atom.GetFormalCharge(),
                atom.GetChiralTag(),
                atom.GetHybridization(),
                atom.GetNumExplicitHs(),
    			atom.GetIsAromatic()]

	features += one_of_k_encoding_unk(atom.GetSymbol(), recognized_elements)
	return np.array(features)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


##########################################################################

def smiles_to_graph(smiles):
	mol = Chem.MolFromSmiles(smiles)
	#mol = Chem.AddHs(mol)
	A = Chem.rdmolops.GetAdjacencyMatrix(mol)

	#TODO: add bond features to feature matrix
	X = np.zeros((mol.GetNumAtoms(), num_atom_features()))

	for i in range(mol.GetNumAtoms()):
		atom = mol.GetAtomWithIdx(i)
		X[i,:] = atom_features(atom)

	return A, X

def pad(A, X, vertex_count):
	new_A = np.zeros((vertex_count, vertex_count))
	new_X = np.zeros((vertex_count, X.shape[1]))

	new_A[:A.shape[0], :A.shape[1]] = A
	new_X[:X.shape[0], :] = X
	return new_A, new_X

def reorder_graph(G):
	dim = G.order()
	T = nx.bfs_tree(G, np.random.randint(dim))
	top = nx.topological_sort(T)
	order = [top.index(i) for i in range(dim)]
	if FLAGS.all_permutations:
		order = np.random.permutation(dim)
	mapping = dict(zip(G.nodes(),order))
	return nx.relabel_nodes(G, mapping)

def read_clintox():
	#hard-coded maximum vertex count specifically for the clintox dataset
	VERTICES = 150

	As = []
	Xs = []
	Cs = []

	with open("clintox.csv", "rb") as f:
		reader = csv.reader(f)
		next(reader, None) #skip header
		for row in reader:
			smiles = row[0]

			if Chem.MolFromSmiles(smiles) is not None:
				A, X = smiles_to_graph(smiles)
				A, X = pad(A, X, VERTICES)
				X = np.hstack((X, np.identity(VERTICES)))
				As.append(A)
				Xs.append(X)
				Cs.append(int(row[1]))

	C = np.zeros((len(Cs), 2))
	for i in range(len(Cs)):
		C[i,Cs[i]] = 1

	return As, Xs, C


def read_mutag():
	#hard-coded maximum vertex count specifically for the mutag dataset
	VERTICES = 30

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
			A, X = pad(A, X, VERTICES)
			X = np.hstack((X, np.identity(VERTICES)))
			As.append(A)
			Xs.append(X)
	return As, Xs, np.stack(Cs)

def read_star():
	dim = FLAGS.max_dim
	batch = FLAGS.training_size

	As = []
	Xs = []

	for i in range(batch):
		G = nx.star_graph(dim-1)
		G = reorder_graph(G)
		A = nx.to_numpy_matrix(G)

		X = np.identity(dim)
		As.append(A)
		Xs.append(X)

	return As, Xs, dim

def read_ring():
	dim = FLAGS.max_dim
	batch = FLAGS.training_size

	As = []
	Xs = []

	for _ in range(batch):
		G = nx.cycle_graph(dim)
		G = reorder_graph(G)
		A = nx.to_numpy_matrix(G)

		X = np.identity(dim)
		As.append(A)
		Xs.append(X)

	return As, Xs, dim

def read_bipartite():
	dim = FLAGS.max_dim
	batch = FLAGS.training_size

	As = []
	Xs = []

	for _ in range(batch):
		G = nx.complete_bipartite_graph(dim - 3, 3)
		G = reorder_graph(G)
		A = nx.to_numpy_matrix(G)

		X = np.identity(dim)
		As.append(A)
		Xs.append(X)

	return As, Xs, dim

def read_ego():
	dim = FLAGS.max_dim
	batch = FLAGS.training_size

	As = []
	Xs = []

	for _ in range(batch):
		G = nx.fast_gnp_random_graph(dim, 0.1)
		A = nx.to_numpy_matrix(G)
		idx = np.random.randint(A.shape[0])
		A[idx, :] = A[:, idx] = 1
		A[idx, idx] = 0

		X = np.identity(dim)
		As.append(A)
		Xs.append(X)

	return As, Xs, dim

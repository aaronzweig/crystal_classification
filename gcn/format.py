# from pynauty import Graph, certificate, autgrp
import numpy as np
from scipy.sparse import csr_matrix, hstack, identity
from rdkit import Chem
import csv
import networkx as nx

recognized_elements = 	['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
						'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
						'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
						'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
						'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

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

	return np.array(one_of_k_encoding_unk(atom.GetSymbol(), recognized_elements))

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC])

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
	A = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO = True)

	A[np.where(A == 3)] = 4
	A[np.where(A == 2)] = 3
	A[np.where(A == 1.5)] = 2

	X = np.zeros((mol.GetNumAtoms(), num_atom_features()))

	for i in range(mol.GetNumAtoms()):
		atom = mol.GetAtomWithIdx(i)
		X[i,:] = atom_features(atom)

	return A, X

def reorder_graph(G):
	dim = G.order()

	rand = np.random.permutation(dim)
	mapping = dict(zip(G.nodes(),rand))
	G = nx.relabel_nodes(G, mapping)

	root = np.random.randint(dim)
	top = [root] + [b for (a,b) in nx.bfs_edges(G, root)]

	order = [top.index(i) for i in range(dim)]
	mapping = dict(zip(G.nodes(),order))

	return nx.relabel_nodes(G, mapping)

def build_molecule_graph(A, X):
    G = nx.from_numpy_matrix(A)
    node_labels = {}
    for i in range(X.shape[0]):
        index = list(X[i]).index(1)
        node_labels[i] = recognized_elements[index]
    nx.set_node_attributes(G, "atom", node_labels)
    G.remove_nodes_from(nx.isolates(G))
    return G

def build_molecule_matrix(G):
	A = nx.to_numpy_matrix(G)
	X = np.zeros((A.shape[0], num_atom_features()))
	for i in range(X.shape[0]):
		X[i, :] = one_of_k_encoding_unk(nx.get_node_attributes(G, "atom")[i], recognized_elements)
	return A, X

def read_clintox_distribution():
	dist = np.zeros(num_atom_features())

	with open("clintox.csv", "rb") as f:
		reader = csv.reader(f)
		next(reader, None) #skip header
		for row in reader:
			smiles = row[0]

			if Chem.MolFromSmiles(smiles) is not None:
				A, X = smiles_to_graph(smiles)
				if A.shape[0] > 11:
					continue

				dist += np.sum(X, 0)

	return dist * 1.0 / np.sum(dist)

def random_indices(k, dist):
	num = dist.size
	samples = np.random.choice(num, k, p = dist)
	indices = zip(range(k), samples)
	return indices

def read_clintox():
	dim = FLAGS.max_dim
	As = []
	Xs = []
	dist = read_clintox_distribution()

	with open("clintox.csv", "rb") as f:
		reader = csv.reader(f)
		next(reader, None) #skip header
		for row in reader:
			smiles = row[0]

			if Chem.MolFromSmiles(smiles) is not None:
				A, X = smiles_to_graph(smiles)
				if A.shape[0] > 11:
					continue

				Xpad = np.zeros((dim, X.shape[1]))
				indices = random_indices(dim, dist)
				for index in indices:
					Xpad[index] = 1
				Xpad[:X.shape[0], :X.shape[1]] = X
				X = np.asarray(np.hstack((Xpad, np.identity(dim))))

				As.append(A)
				Xs.append(X)

	return As, Xs, dim


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

def read_zinc():
	dim = 20

	As = []
	Xs = []

	count = 0
	with open("50_k.smi") as f:
		for line in f:
			smiles = line.split()[0]
			if Chem.MolFromSmiles(smiles) is not None:
				A, X = smiles_to_graph(smiles)
				G = build_molecule_graph(A, X)
				G = reorder_graph(G)
				A, X = build_molecule_matrix(G)


        		Apad = np.zeros((dim, dim))
        		Apad[:A.shape[0], :A.shape[1]] = A

        		Xpad = np.zeros((dim, X.shape[1]))
        		Xpad[:, -1] = 1 #remaining nodes are deemed unknown
        		Xpad[:X.shape[0], :X.shape[1]] = X

        		As.append(Apad)
        		Xs.append(Xpad)

        		count += 1
        		if count >= FLAGS.training_size:
        			break

	return As, Xs, dim
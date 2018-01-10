import pickle
from collections import Counter
import numpy as np

dataset = ['mutag.graph', 'ptc.graph', 'enzymes.graph', 'proteins.graph', 'nci1.graph', 'nci109.graph', 'collab.graph', 'imdb_action_romance.graph', 'reddit_iama_askreddit_atheism_trollx.graph', 'reddit_multi_5K.graph', 'reddit_subreddit_10K.graph']

def load_data(ds_name):
    f = open(ds_name, "r")
    data = pickle.load(f)
    graph_data = data["graph"]
    labels = data["labels"]
    labels  = np.array(labels, dtype = np.float)
    return graph_data, labels

for ds in dataset:
    print "Reading dataset ", ds
    graph, labels = load_data(ds)
    if ds == 'proteins.graph':
        labels = labels[0]
    print "Dataset: %s length: %s label distribution: %s"%(ds, len(graph), Counter(labels))
    avg_nodes = []
    for gidx, nodes in graph.iteritems():
        avg_nodes.append(len(nodes))
    print "Avg #nodes: %s Median #nodes: %s Max #nodes: %s Min #nodes: %s"%(np.mean(avg_nodes), np.median(avg_nodes), max(avg_nodes), min(avg_nodes))





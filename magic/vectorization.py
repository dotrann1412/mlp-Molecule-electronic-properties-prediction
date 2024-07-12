from .wl import str2method
from .smiles import smiles2graph 
import numpy as np 

class GraphVectorizer(object):
    def __init__(self, label_method=None, num_iter=None, smiles=True):

        self.unique_labels = []
        self.num_iter = num_iter
        self.label_method = label_method

        if isinstance(self.label_method, str):
            self.label_method = str2method(self.label_method)

        self.smiles = smiles

    def fit(self, X):
        if self.smiles: 
            X = smiles2graph(X)

        print(f"Vectorizing {len(X)} graphs")

        for graph in X:
            counter = self.label_method(*graph).to_counter(self.num_iter)
            self.unique_labels += list(counter.keys())
            self.unique_labels = list(set(self.unique_labels))

        return self

    def vectorize(self, graph):
        if self.smiles: 
            graph = smiles2graph(graph)

        counter = self.label_method(*graph).to_counter(self.num_iter)
        x = []

        for label in self.unique_labels:
            x.append(counter[label] if label in counter else 0)

        return x

    def transform(self, graphs: list):
        X = np.zeros((len(graphs), len(self.unique_labels)))

        for i,graph in enumerate(graphs):
            x = self.vectorize(graph)
            X[i] += np.array(x)

        return X
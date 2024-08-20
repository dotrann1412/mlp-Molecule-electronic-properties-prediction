from .wl import str2method, WL, load as load_wl
from .smiles import smiles2graph 
import numpy as np 

class GraphVectorizer(object):
    def __init__(self, label_method=None, num_iter=None, smiles=True):

        self.unique_labels = []
        self.num_iter = num_iter
        self.label_method = label_method
        self.smiles = smiles
        
    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "num_iter": self.num_iter,
            "smiles": self.smiles,
            "label_method": self.label_method,
            "unique_labels": self.unique_labels
        }
        
    @classmethod
    def from_json(cls, data):
        x = cls(
            data["label_method"],
            data["num_iter"],
            data["smiles"]
        )

        x.unique_labels = data["unique_labels"]
        return x

    def fit(self, X):
        if self.smiles: 
            X = smiles2graph(X)

        for graph in X:
            counter = str2method(self.label_method)(*graph).to_counter(self.num_iter)
            self.unique_labels += list(counter.keys())
            self.unique_labels = list(set(self.unique_labels))

        return self

    def vectorize(self, graph):
        if self.smiles: 
            graph = smiles2graph(graph)

        counter = str2method(self.label_method)(*graph).to_counter(self.num_iter)
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
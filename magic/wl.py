from zlib import crc32
from collections import Counter
from itertools import chain

class WL:
    """
    Base class for Weisfeiler-Lehman labelling graph
    """
    def __init__(self, nodes_feat, adj, *_):

        """initiate labels for each node by hashing a list of it properties."""
        self.adj = adj

        # atom_labels for storing all labels lists across all iterations
        self.atom_labels = [[
            self.hash(feat) for feat in nodes_feat]]

    def to_json(self):
        return {
            "atom_labels": self.atom_labels,
            "adj": self.adj,
            "name": self.__class__.__name__
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            data["atom_labels"],
            data["adj"]
        )

    def hash(self,l):
        """
        Return an integer from a list by hashing it
        """
        strl = "".join([str(i) for i in l])
        hash_int = crc32(strl.encode("utf8")) & 0xffffffff
        return hash_int

    def get_adj(self,atom_idx):
        """
        Return adjcent atoms' indices
        """
        return self.adj[atom_idx]

    def relabelling_nodes(self):
        atom_labels = self.atom_labels[-1]
        new_atomic_labels = []
        """
        Essentially perform one iteration of WL algorithm.
        After this function is called, a new set of labels will
            be appended to self.atom_labels
        """
        for a1,atom_label in enumerate(atom_labels):
            # get adjacent atoms' indices
            adj_atoms_indices = self.get_adj(a1)

            # put adjacent atoms' labels into a list and sort it 
            M = [atom_labels[idx] for idx in adj_atoms_indices]
            M = sorted(M)

            # insert label of the main atom into the beginning
            M.insert(0,atom_labels[a1])

            # hash and insert the new label into the list
            new_atomic_labels.append(
                self.hash(M))

        self.atom_labels.append(new_atomic_labels)

    def to_counter(self, *args, **kwargs):
        """
        Return a Counter object of all labels after num_iters iterations
        """
        raise NotImplementedError

class WLSubtree(WL):
    """
    Class for Weisfeiler-Lehman Subtree of Atom-based method
    """
    def __init__(self, nodes, adj, *_):
        super().__init__(nodes, adj)
        
    def to_json(self):
        x = super().to_json()
        x['name'] = self.__class__.__name__
        return x

    def to_counter(self,num_iters):
        for i in range(num_iters):
            self.relabelling_nodes()

        atom_labels = chain(*self.atom_labels)
        return Counter(atom_labels)
        

class WLEdge(WL):
    def __init__(self, nodes, adj, edges, edges_feats, *_):
        super().__init__(nodes, adj)

        self.edges = edges
        self.edges_feats = edges_feats

        self.edge_labels = []
        self.relabelling_edges()
        
    def to_json(self):
        return {
            "nodes": self.atom_labels,
            "adj": self.adj,
            "edges": self.edges,
            "edges_feats": self.edges_feats,
            "name": self.__class__.__name__
        }
            
        
    @classmethod
    def from_json(cls, data):
        return cls(
            data["nodes"],
            data["adj"],
            data["edges"],
            data["edges_feats"]
        )

    def relabelling_edges(self):
        edge_labels = []
        atom_labels = self.atom_labels[-1]

        for i,edge in enumerate(self.edges):
            a1_idx,a2_idx = edge
            M = sorted(
                [atom_labels[idx] for idx in [a1_idx,a2_idx]])
            M += self.edges_feats[i]
            edge_labels.append(self.hash(M))

        self.edge_labels.append(edge_labels)

    def to_counter(self,num_iters):
        for i in range(num_iters):
            self.relabelling_nodes()
            self.relabelling_edges()

        edge_labels = chain(*self.edge_labels)
        return Counter(edge_labels)

class WLShortestPath(WL):
    def __init__(self, nodes, adj, sp_dists, *_):
        super().__init__(nodes,adj)

        self.num_nodes = len(nodes)
        self.adj = adj

        self.sp_dists = sp_dists

        self.path_labels = []
        self.relabelling_path()
        
    def to_json(self):
        return {
            "nodes": self.atom_labels,
            "adj": self.adj,
            "sp_dists": self.sp_dists,
            "name": self.__class__.__name__
        }
        
    @classmethod
    def from_json(cls, data):
        return cls(
            data["nodes"],
            data["adj"],
            data["sp_dists"]
        )

    def relabelling_path(self):
        path_labels = []

        atom_labels = self.atom_labels[-1]

        for i,label in enumerate(atom_labels):
            for atom_idx, path_len in enumerate(self.sp_dists[i]):
                if i > atom_idx: continue 
                M = sorted([
                    label, atom_labels[atom_idx]])
                M.append(path_len)
                path_labels.append(self.hash(M))

        self.path_labels.append(path_labels)
        
    def to_counter(self,num_iters):
        for i in range(num_iters):
            self.relabelling_nodes()
            self.relabelling_path()

        path_labels = chain(*self.path_labels)
        return Counter(path_labels)

def str2method(s):
    if s == "WLSubtree":
        return WLSubtree
    elif s == "WLEdge" :
        return WLEdge
    elif s == "WLShortestPath" :
        return WLShortestPath
    else:
        raise ValueError("Invalid method")

def load(data):
    if data["name"] == "WLSubtree":
        return WLSubtree.from_json(data)
    elif data["name"] == "WLEdge":
        return WLEdge.from_json(data)
    elif data["name"] == "WLShortestPath":
        return WLShortestPath.from_json(data)
    else:
        raise ValueError("Invalid method")

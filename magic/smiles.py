"""
Original code at:
https://github.com/snap-stanford/ogb/blob/master/ogb/utils/mol.py
"""

import time
import numpy as np
from rdkit import Chem
import numpy as np
import rdkit

# allowable multiple choice node and edge features 
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l,e):
    try:
        #return the index of the property in the allowable properties list
        return l.index(e)
    except:
        #else return the last index, which is misc
        return len(l) - 1

def featurize_atom(atom,minimal = False):
    atom_feature = safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum())

    if not minimal:
        atom_feature = [atom_feature] + [
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature

def featurize_bond(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature


class Dijkstra:
    """
    written based on pseudocode from wikipedia page on Dijkstra algorithm
    https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
    """
    def __init__(self,num_nodes,adj):
        self.num_nodes = num_nodes
        self.adj = adj

    def find(self,s_idx):
        """
        Determine the shortest path between the index of source atom (s_idx) and 
        every other atoms in the molecule.
        Args:
        + atom_s (AtomNode): atom source
        Return:
        + dist (dictionary): with keys are other atoms and values are shortest distance
        """
        dist = [np.inf]*self.num_nodes
        dist[s_idx] = 0

        prev = [None]*self.num_nodes
        Q = []

        for vertex_idx in range(self.num_nodes):
            Q.append(vertex_idx)
        
        while len(Q) > 0:
            """
            u is main node
            v is neighbor nodes of main node
            """

            mini_dist = {v:dist[v] for v in Q}
            min_dist = min(mini_dist.values())
            vertices_min_d = [k for k, v in mini_dist.items() 
                              if v == min_dist]

            for u in vertices_min_d:
                Q.remove(u) 
                for v in self.adj[u]:
                    alt = dist[u] + 1

                    if alt < dist[v]:
                        dist[v], prev[v] = alt, u

        return dist

def smiles2graph(smiles, sp=False, minimal=False):
    if isinstance(smiles, list):
        graphs = []

        for s in smiles:
            graph = smiles2graph(s, sp=sp)
            graphs.append(graph)

        return graphs
            
    elif isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)

        node_feat  = []

        for atom in mol.GetAtoms():
            node_feat.append(featurize_atom(atom, minimal=minimal))
            
        edges_list = []
        adj_list = [[] for node in node_feat]
        edges_feat = []

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx() 
            a2 = bond.GetEndAtomIdx() 

            edges_list.append((a1,a2))

            adj_list[a1].append(a2)
            adj_list[a2].append(a1)

            edges_feat.append(featurize_bond(bond))

        if sp:
            sp_dists = []
            num_nodes = len(node_feat)
            sp_algo = Dijkstra(num_nodes,adj_list)

            for i in range(num_nodes):
                sp_dists.append(sp_algo.find(i))

            return node_feat, adj_list,edges_list, edges_feat, sp_dists

        else:
            return node_feat, adj_list,edges_list, edges_feat

    else: 
        raise Exception("Expect list or str, has {}".format(type(smiles)))
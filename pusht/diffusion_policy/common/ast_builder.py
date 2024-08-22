import ring
import numpy as np

import dgl
import networkx as nx
from sklearn.preprocessing import OneHotEncoder

edge_types = {k:v for (v, k) in enumerate(["self", "arg", "arg1", "arg2"])}

"""
A class that can take an LTL formula and generate the Abstract Syntax Tree (AST) of it. This
code can generate trees in either Networkx or DGL formats. And uses caching to remember recently
generated trees.
"""
class ASTBuilder(object):
    def __init__(self, propositions):
        super(ASTBuilder, self).__init__()

        self.props = propositions

        terminals = list(set(list(['True', 'False'] + self.props)))
        ## Pad terminals with dummy propositions to get a fixed encoding size
        for i in range(15 - len(terminals)):
            terminals.append("dummy_"+str(i))

        self._enc = OneHotEncoder(handle_unknown='ignore', dtype=int)
        self._enc.fit([['next'], ['until'], ['and'], ['or'], ['eventually'],
            ['always'], ['not']] + np.array(terminals).reshape((-1, 1)).tolist())

    # To make the caching work.
    def __ring_key__(self):
        return "ASTBuilder"

    @ring.lru(maxsize=30000)
    def __call__(self, formula, dim_is_root=1, library="dgl"):
        nxg = self._to_graph(formula)
        # nx.set_node_attributes(nxg, 0., "is_root")
        # nxg.nodes[0]["is_root"] = 1.
        nx.set_node_attributes(nxg, np.zeros(dim_is_root, dtype="float32"), "is_root")
        nxg.nodes[0]["is_root"] = np.ones(dim_is_root, dtype="float32")
        if (library == "networkx"): return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        g = dgl.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"], device='cuda')
        # g = dgl.DGLGraph()
        # g.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)
        return g

    def _one_hot(self, token):
        return self._enc.transform([[token]])[0][0].toarray()


    def _get_edge_type(self, operator, parameter_num=None):
        operator = operator.lower()
        if (operator in ["next", "until", "and", "or"]):
            # Uncomment to make "and" and "or" permutation invariant
            # parameter_num = 1 if operator in ["and", "or"] else operator

            return edge_types[operator + f"_{parameter_num}"]

        return edge_types[operator]

    # A helper function that recursively builds up the AST of the LTL formula
    @ring.lru(maxsize=60000) # Caching the formula->tree pairs in a Last Recently Used fashion
    def _to_graph(self, formula, shift=0):
        head = formula[0]
        rest = formula[1:]
        nxg  = nx.DiGraph()

        if head in ["until", "and", "or"]:
            nxg.add_node(shift, feat=self._one_hot(head), token=head)
            nxg.add_edge(shift, shift, type=self._get_edge_type("self"))

            l = self._to_graph(rest[0], shift+1) # build the left subtree
            nxg = nx.compose(nxg, l) # combine the left subtree with the current tree
            nxg.add_edge(shift+1, shift, type=self._get_edge_type("arg1")) # connect the current node to the root of the left subtree

            index = nxg.number_of_nodes()
            r = self._to_graph(rest[1], shift+index) # build the left subtree
            nxg = nx.compose(nxg, r) # combine the left subtree with the current tree
            nxg.add_edge(shift+index, shift, type=self._get_edge_type("arg2"))
            # if head in ["next", "until"]:
            #     nxg.add_edge(1, index, type=self.ASSYM_EDGE) # impose order on the operands of an assymetric operator

            return nxg

        if head in ["next", "eventually", "always", "not"]:
            nxg.add_node(shift, feat=self._one_hot(head), token=head)
            nxg.add_edge(shift, shift, type=self._get_edge_type("self"))

            l = self._to_graph(rest[0], shift+1) # build the left subtree
            nxg = nx.compose(nxg, l) # combine the left subtree with the current tree
            nxg.add_edge(shift+1, shift, type=self._get_edge_type("arg")) # connect the current node to the root of the left subtree

            return nxg

        if formula in ["True", "False"]:
            nxg.add_node(shift, feat=self.vocab._one_hot(formula), token=formula)
            nxg.add_edge(shift, shift, type=self._get_edge_type("self"))

            return nxg

        if formula in self.props:
            nxg.add_node(shift, feat=self._one_hot(formula.replace("'",'')), token=formula)
            nxg.add_edge(shift, shift, type=self._get_edge_type("self"))

            return nxg


        assert False, "Format error in ast_builder.ASTBuilder._to_graph()"

        return None

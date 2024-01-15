import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import scib
# from scib.metrics import lisi_graph_py
import anndata as ad


# ----------------------------------------------------------------------------- #
#                                                                               #
#                functions for processing connected components                  #
#                                                                               #
# ----------------------------------------------------------------------------- #


# --- helper to count the number of unique subjects given list of nodes --- #
# inputs: graph, iterable nodes
# returns: number of unique subjects
def count_unique_subjects(graph, nodes):
    unique_subjects_set = set()

    for node in nodes:
        unique_subjects_set.add(graph.nodes[node]["subject_id"]) # can even pass in desired column name (str)

    return len(unique_subjects_set) 

# --- list of remove/keep conditions for connected components --- #
# - TRUE if connected component has at least n_nodes nodes
def has_atleast_node(graph, cc, n_nodes=1):
    return len(cc) >= n_nodes

# - TRUE if connected component has at least n_subject unique subjects
def has_atleast_subject(graph, cc, n_subjects=1):
    return count_unique_subjects(graph, cc) >= n_subjects

# - TRUE if connected component has at least n_nodes nodes and n_subjects unique subjects
def has_atleast_node_subject(graph, cc, n_nodes=1, n_subjects=1):
    return has_atleast_node(graph, cc, n_nodes=n_nodes) and has_atleast_subject(graph, cc, n_subjects=n_subjects) # wrapper


# ----------------------------------------------------------------------------- #
#                                                                               #
#            functions for processing nodes in connected components             #
#                                                                               #
# ----------------------------------------------------------------------------- #

# --- manual statistics of nodes in cc --- #
# counts number of unique subjects among a node's neighbors
def count_unique_subjects_among_neighbors(subgraph, node):
    unique_neighbors_set = set()

    for neighbor in subgraph.neighbors(node):
        unique_neighbors_set.add(subgraph.nodes[neighbor]["subject_id"])

    return len(unique_neighbors_set)

# counts number of neighbors that come from a different subject from itself
def count_different_subject_neighbors(subgraph, node):
    count = 0

    node_subject_id = subgraph.nodes[node]["subject_id"]

    for neighbor in subgraph.neighbors(node):
        if subgraph.nodes[neighbor]["subject_id"] != node_subject_id:
            count += 1

    return count
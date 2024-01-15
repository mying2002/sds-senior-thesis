import numpy as np
import pandas as pd
import networkx as nx
from heapq import heappop, heappush


def compute_lisi_metrics(subgraph, dist_attr="weight", label_attr="subject_id", k=None, perplexity=None):
    """
    @param subgraph: graph for which we are computing lisi; a connected component
    @param dist_attr: name of edge attribute used for distance
    @param label_attr: name of node attribute used as labels
    @param k: how many nearest neighbors to use? 

    RETURNS: dict{node : lisi_score} for integration with df AND dataframe with node, lisi as columns; original node names
    """

    # default values
    if k == None:
        k = min(len(subgraph.nodes) - 1, 40)
    if perplexity == None:
        perplexity = max(k/3, 3) # no smaller than 3 (smallest CC has 4 nodes)
    
    # checks
    assert k < len(subgraph.nodes) # strictly less
    assert perplexity <= k # optional?

    # TO IMPLEMENT: instead of assert, maybe auto assign k/perp if they aren't reasonable

    # --- main set up --- #
    # [OLD KNN - TOO SLOW FOR BIG GRAPHS!] compute distances
    # subgraph_distances_dict = dict(nx.all_pairs_dijkstra_path_length(subgraph, weight=dist_attr)) 

    # mapping between node names and indexes (ie. 0 to n-1)
    node_list = list(subgraph.nodes) # keep this order!!
    index_list = [i for i in range(len(node_list))]
    node_to_index_dict = dict(zip(node_list, index_list))
    index_to_node_dict = dict(zip(index_list, node_list))

    # store knn + knn distances for each node
    subgraph_knn = pd.DataFrame()
    subgraph_knn_distances = pd.DataFrame()

    # compute knn + knn distances per node; concat to dataframes
    # same order as node_list 
    for node in subgraph.nodes:
        # [OLD KNN - TOO SLOW FOR BIG GRAPHS!]
        # distances_dict = subgraph_distances_dict[node]
        # sorted_distances_tuple_list = sorted(distances_dict.items(), key=lambda x:x[1]) # sort by distances; includes itself
        # closest_k_neighbors = [neighbor for (neighbor, distance) in sorted_distances_tuple_list[1:(k+1)]] # don't include item 0 (itself)
        # closest_k_neighbors_distances = [distance for (neighbor, distance) in sorted_distances_tuple_list[1:(k+1)]]

        # [NEW KNN - MANUAL DIJKSTRA'S]
        knn_dict = compute_graph_knn(subgraph, node, k=k, weight_attr=dist_attr, ignore_labels=None)
        closest_k_neighbors = list(knn_dict.keys())
        closest_k_neighbors_distances = list(knn_dict.values())

        # - indexes of nearest neighbors
        # print(k)
        knn_row_df = pd.DataFrame([closest_k_neighbors]) # to be concatenated
        knn_row_df.columns = [f"neighbor_{i:d}" for i in range(k)]
        knn_row_df.index = [node]

        subgraph_knn = pd.concat([subgraph_knn, knn_row_df])

        # - distances to nearest neighbors
        knn_distances_row_df = pd.DataFrame([closest_k_neighbors_distances])
        knn_distances_row_df.columns = [f"neighbor_{i:d}" for i in range(k)]
        knn_distances_row_df.index = [node]

        subgraph_knn_distances = pd.concat([subgraph_knn_distances, knn_distances_row_df])

    
    # prep inputs for helper function
    dist_matrix = subgraph_knn_distances.to_numpy() # loses index and column names, but preserves order

    node_to_index_func = np.vectorize(node_to_index_dict.get)
    knn_nodes = subgraph_knn.to_numpy()
    knn_indexes = node_to_index_func(knn_nodes) 

    node_subject_id_dict = nx.get_node_attributes(subgraph, label_attr)
    batch_labels = list(map(node_subject_id_dict.get, node_list)) 
    # https://stackoverflow.com/questions/71690493/numpy-alternative-to-pd-factorize # convert subject ids to ints
    batch_labels_int = np.unique(batch_labels, return_inverse=True)[1]

    # call helper function and format result 
    simpson_metrics = compute_simpson_index(D=dist_matrix, knn_idx=knn_indexes, batch_labels=batch_labels_int, perplexity=perplexity)
    lisi_metrics = 1/simpson_metrics

    node_to_lisi_score_dict = dict(zip(node_list, lisi_metrics))
    lisi_df = pd.DataFrame({"node" : node_list, "lisi_score" : lisi_metrics})

    return node_to_lisi_score_dict, lisi_df


# measures "effective" number of labels
def compute_naive_simpson_index(subgraph, label_attr="subject_id"):
    # need a multiset
    label_count_dict = dict()
    for node in subgraph.nodes():
        node_label = subgraph.nodes[node][label_attr]
        if node_label in label_count_dict:
            label_count_dict[node_label] = label_count_dict[node_label] + 1
        else:
            label_count_dict[node_label] = 1
    
    numerator_sum = 0
    N = 0

    # --- assumes n(n-1) for finite sample
    # for value in label_count_dict.values():
    #     numerator_sum += value*(value - 1)
    #     N += value

    # simpson_index = numerator_sum / (N * (N-1))

    # --- assumes n^2 (with replacement)
    for value in label_count_dict.values():
        numerator_sum += value**2
        N += value

    simpson_index = numerator_sum / (N**2)

    return 1 / simpson_index



# ----- manual k nearest neighbors algorithm ----- #
# subgraph: graph to compute knn on
# node: node of interest
# weight_attr: edge label used for edge length calculation
# ignore_labels: list of tuples of (node_attr, value) describing nodes to ignore [ie. healthy controls]
# returns list of k neighbors (subgraph node label) and list of k corresponding distances. 
def compute_graph_knn(subgraph, node, k, weight_attr="weight", ignore_labels=None):
    assert nx.is_connected(subgraph)
    assert k < len(subgraph.nodes)
    assert node in subgraph.nodes

    minheap = [] # heap storing (distance from source node, node name); sorts by first entry
    # heappush, heappop
    knn_dict = {} # dict storing finalized (node, dist) to the knn of 'node'; finalized. 
    seen = {} # dict storing (node, dist) of seen nodes; stores current shortest distance!

    seen[node] = 0
    for n in subgraph.neighbors(node):
        node_to_n_weight = subgraph[node][n][weight_attr]
        seen[n] = node_to_n_weight
        heappush(minheap, (node_to_n_weight, n))

    while len(knn_dict) < k:
        # 
        dist, u = heappop(minheap)

        # if u already in dict
        if u in knn_dict:
            # error check, but don't really do anything
            if dist < knn_dict[u]:
                print("Error catch: found a negative path.")
        # if not in dict
        else:
            # add node to dict as one of the knn.
            knn_dict[u] = dist

            # update seen nodes; update heap only if dist decreased. 
            for v in subgraph.neighbors(u):
                # grab uv edge length
                uv_dist = subgraph[u][v][weight_attr]

                # compute distance to v going through u
                dist_through_u_to_v = knn_dict[u] + uv_dist

                # update heap, seen if needed; don't update if already seen and doesn't matter. 
                if v in seen:
                    if dist_through_u_to_v < seen[v]:
                        seen[v] = dist_through_u_to_v
                        heappush(minheap, (dist_through_u_to_v, v))
                else:
                    seen[v] = dist_through_u_to_v
                    heappush(minheap, (dist_through_u_to_v, v))
                
    
    # print(knn_dict)
                

    return knn_dict



# ----------------------------------------- #
#               from scib docs              #
# ----------------------------------------- #


def compute_simpson_index(
    D=None, knn_idx=None, batch_labels=None, n_batches=None, perplexity=15, tol=1e-5
):
    """
    Simpson index of batch labels subset by group.

    :param D: distance matrix ``n_cells x n_nearest_neighbors``         # assumes nodes are in order, 0 through n-1
    :param knn_idx: index of ``n_nearest_neighbors`` of each cell       # indices should match ^ and be 0 through n-1
    :param batch_labels: a vector of length n_cells with batch info     # integer labele from 0, please!
    :param n_batches: number of unique batch labels
    :param perplexity: effective neighborhood size
    :param tol: a tolerance for testing effective neighborhood size
    :returns: the simpson index for the neighborhood of each cell
    """
    n = D.shape[0]
    P = np.zeros(D.shape[1])
    simpson = np.zeros(n)
    logU = np.log(perplexity)

    # loop over all cells
    for i in np.arange(0, n, 1):
        beta = 1
        # negative infinity
        betamin = -np.inf
        # positive infinity
        betamax = np.inf
        # get active row of D
        D_act = D[i, :]
        H, P = Hbeta(D_act, beta)
        Hdiff = H - logU
        tries = 0
        # first get neighbor probabilities
        while np.logical_and(np.abs(Hdiff) > tol, tries < 50):
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            H, P = Hbeta(D_act, beta)
            Hdiff = H - logU
            tries += 1
        
        # print(beta)

        if H == 0:
            simpson[i] = -1
            continue

            # then compute Simpson's Index
        non_nan_knn = knn_idx[i][np.invert(np.isnan(knn_idx[i]))].astype("int")
        batch = batch_labels[non_nan_knn]
        # convertToOneHot omits all nan entries.
        # Therefore, we run into errors in np.matmul.
        if len(batch) == len(P):
            B = convert_to_one_hot(batch, n_batches)
            sumP = np.matmul(P, B)  # sum P per batch
            simpson[i] = np.dot(sumP, sumP)  # sum squares
        else:  # assign worst possible score
            simpson[i] = 1

    return simpson




def Hbeta(D_row, beta):
    """
    Helper function for simpson index computation
    """
    P = np.exp(-D_row * beta)
    sumP = np.nansum(P)
    if sumP == 0:
        H = 0
        P = np.zeros(len(D_row))
    else:
        H = np.log(sumP) + beta * np.nansum(D_row * P) / sumP
        P /= sumP
    return H, P


def convert_to_one_hot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output 2-D array of one-hot vectors,
    where an i'th input value of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:

    .. code-block:: python

        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print(one_hot_v)

    .. code-block::

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    # assert isinstance(vector, np.ndarray)
    # assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector) + 1
    # else:
    #    assert num_classes > 0
    #    assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)
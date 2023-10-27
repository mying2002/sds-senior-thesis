# ----------------------------------------- #
#               from scib docs              #
# ----------------------------------------- #
import numpy as np


def compute_simpson_index(
    D=None, knn_idx=None, batch_labels=None, n_batches=None, perplexity=15, tol=1e-5
):
    """
    Simpson index of batch labels subset by group.

    :param D: distance matrix ``n_cells x n_nearest_neighbors``
    :param knn_idx: index of ``n_nearest_neighbors`` of each cell
    :param batch_labels: a vector of length n_cells with batch info
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
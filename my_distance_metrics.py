import numpy as np
from Levenshtein import distance


def hamming_distance_norm(str1, str2):
    assert len(str1) == len(str2)

    hamming_dist = sum(c1 != c2 for c1, c2 in zip(str1, str2)) / len(str1)

    # matching strings; set epsilon dist
    epsilon = 1e-5
    if hamming_dist == 0:
        hamming_dist += epsilon

    return hamming_dist

# case 1: all operations cost 1.
# normalizing: (inserts/deletions + 2 * substitutions) / (len1 + len2) --> ensures normalized in [0,1]
    # L.distance computes inserts/deletions + 2 * substitutions for us
# score cutoff = cutoff for edge
# Q: do we want substitutions to count as 1 mismatch or 2? if just 1, then complete mismatches will be 0.5 
def levenshtein_distance_norm(str1, str2, weights=(1,1,2), edge_cutoff=None):
    if edge_cutoff == None:
        leven_dist = distance(str1, str2, weights=weights)
    else:
        score_cutoff = edge_cutoff * (len(str1) + len(str2)) # scale up; pass into levenshtein distance
        leven_dist = distance(str1, str2, weights=weights, score_cutoff=score_cutoff) # gives int number of effective changes. 

    # normalize
    leven_dist /= (len(str1) + len(str2))

    # if perfect match, assign tiny distance for edge
    epsilon = 1e-5 
    if leven_dist == 0:
        leven_dist = epsilon

    return leven_dist 



# takes in aa_weights_dict (MAYBE WANT 2D DF?) to represent custom weights for certain replacements
# should be relative to 1 for an insertion/deletion
def edit_distance_weighted(str1, str2, aa_replacement_dict):
    # want minimum distance in DP

    return
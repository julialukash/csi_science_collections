import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from numpy.linalg import norm as euclidean_norm
from scipy.stats import entropy

def kl_dist(p, q):
    return entropy(p, q)

def kl_sym_dist(p, q):
    return 0.5 * (entropy(p, q) + entropy(q, p)) 

def jaccard_dist(p, q, top_words_count=15):
    c1_top_words = p.sort_values()[::-1][0:top_words_count]
    c2_top_words = q.sort_values()[::-1][0:top_words_count]
    return 1 - 1.0 * len(c1_top_words.index.intersection(c2_top_words.index)) / len(c1_top_words.index.union(c2_top_words.index))

def euc_dist(p, q):
    return euclidean_norm(p - q)

def cos_dist(p, q):
    p = p.values.reshape(1, -1)
    q = q.values.reshape(1, -1)
    return cosine_distances(p, q)[0][0]

def hellinger_dist(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
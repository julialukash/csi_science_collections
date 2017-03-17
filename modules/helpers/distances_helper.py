import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from numpy.linalg import norm as euclidean_norm
from scipy.stats import entropy

def kl_dist(p, q):
    eps = 1e-30    
    p = p.copy()
    q = q.copy()
    p[p == 0] = eps
    q[q == 0] = eps
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


def calculate_distances(dist_fun, _phi, _phi_other, _debug_print=False):
    if _debug_print:
        print '[{}] take_distances between {} columns and {} columns'.format(datetime.now(), len(_phi.columns), len(_phi_other.columns))
    distances = pd.DataFrame(0, index = _phi.columns, columns=_phi_other.columns)
    for idx, col in enumerate(_phi.columns):
        if _debug_print and idx % 20 == 0:
            print '[{}] column {} / {}'.format(datetime.now(), idx, len(_phi.columns))
        for idx_other, col_other in enumerate(_phi_other.columns):
            distance = dist_fun(_phi[col], _phi_other[col_other])
            distances.iloc[idx, idx_other] = distance
    return distances

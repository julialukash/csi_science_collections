from os import path, mkdir
from sys import stdout
from time import sleep
from scipy.optimize import linear_sum_assignment

import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import distances_helper as dh 
import build_convex_hull_helper as bchh

def dynamic_print(data):
    stdout.write("\r\x1b[K"+data.__str__())
    stdout.flush()

def get_cost_matrix(original_distance, not_original_column_fine_right, not_test_column_fine_down, new_new_fine_corner=1e-5):
    default_empty_value, eps = 100, 1e-5
    distance = original_distance.copy()
    # cut by threshold
    distance[distance > not_original_column_fine_right] = default_empty_value
    test_columns_count, original_columns_count =  distance.shape
    right_matrix = pd.DataFrame(np.identity(test_columns_count) * not_original_column_fine_right, index = distance.index)
    down_matrix = pd.DataFrame(np.identity(original_columns_count) * not_test_column_fine_down, columns = distance.columns)
    corner_matrix = pd.DataFrame(new_new_fine_corner, index = down_matrix.index, columns = right_matrix.columns)
    expanded_down_matrix = pd.concat([down_matrix, corner_matrix], axis=1)
    expanded_matrix =  pd.concat([distance, right_matrix], axis=1)
    expanded_matrix =  pd.concat([expanded_matrix, expanded_down_matrix], axis=0)
    expanded_matrix[np.abs(expanded_matrix) < eps] = default_empty_value
    return expanded_matrix

def indices_to_result(cost_matrix_not_expanded, row_ind, col_ind):
    test_columns_count, original_columns_count =  cost_matrix_not_expanded.shape
    indices = zip(row_ind, col_ind)
    indices = [val for val in indices if val[0] < test_columns_count and val[1] < original_columns_count]
    indices = [(cost_matrix_not_expanded.index[val[0]], cost_matrix_not_expanded.columns[val[1]],
                val[0], val[1], cost_matrix_not_expanded.iloc[val[0], val[1]]) for val in indices]
    return indices

def get_hungarian_alg_result(cost_matrix, fine):
    expanded_cost = get_cost_matrix(cost_matrix, fine, fine, new_new_fine_corner=1e-5)
    row_ind, col_ind = linear_sum_assignment(expanded_cost)
    res = indices_to_result(cost_matrix, row_ind, col_ind)
    return res, expanded_cost, row_ind, col_ind

def get_test_to_original_result(phi_test, phi_original, dist_fn, thresholds):
    cost = dh.calculate_distances(dist_fn, phi_test, phi_original)
    results = {}
    for th in thresholds:
        res, expanded_cost, row_ind, col_ind = get_hungarian_alg_result(cost, th)
        dynamic_print('Dist fn = {}, Processed th = {}, original columns count = {}    '.format(dist_fn, th, len(res)))
        results[th] = res
    return results

def get_test_to_original_result_different_distances(phi_test, phi_original, 
     distances=[dh.jaccard_dist, dh.cos_dist, dh.hellinger_dist, dh.kl_dist, dh.kl_sym_dist],
     thresholds=np.arange(0.05, 1, 0.05)):
    different_distances = {}
    kl_thresholds = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    for dist_fn in distances:
        if dist_fn is dh.kl_dist or dist_fn is dh.kl_sym_dist:
            results = get_test_to_original_result(phi_test, phi_original, dist_fn, kl_thresholds)
        else:
            results = get_test_to_original_result(phi_test, phi_original, dist_fn, thresholds)
        threshold_and_original_columns_count = sorted([(key, len(results[key])) for key in results.iterkeys()])
        different_distances[dist_fn] = (threshold_and_original_columns_count, results)
    return different_distances

#============================================================= opt res =================================================================

def get_test_to_original_opt_result_different_distances(phi_test, phi_original, 
            distances=[dh.jaccard_dist, dh.cos_dist, dh.hellinger_dist, dh.kl_dist, dh.kl_sym_dist], 
            thresholds=np.arange(0.05, 1.05, 0.05)):
    different_distances = {}
    kl_thresholds = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    for dist_fn in distances:
        if dist_fn is dh.kl_dist or dist_fn is dh.kl_sym_dist:
            results = get_test_to_original_opt_result(phi_test, phi_original, dist_fn, kl_thresholds)
        else:
            results = get_test_to_original_opt_result(phi_test, phi_original, dist_fn, thresholds)
        threshold_and_original_columns_count = sorted([(key, len(results[key])) for key in results.iterkeys()])
        different_distances[dist_fn] = (threshold_and_original_columns_count, results)
    return different_distances

def get_test_to_original_opt_result(phi_test, phi_original, dist_fn, thresholds):
    cost = get_cost_opt_matrix(dist_fn, phi_test, phi_original)
    results = {}
    for th in thresholds:
        res, expanded_cost, row_ind, col_ind = get_hungarian_alg_result(cost, th)
        dynamic_print('Dist fn = {}, Processed th = {}, original columns count = {}    '.format(dist_fn, th, len(res)))
        results[th] = res
    return results

def get_cost_opt_matrix(dist_fn, phi_test, phi_original):
    N_CLOSEST_TOPICS = 15
    tmp_distances = dh.calculate_distances(dist_fn, phi_test, phi_original)
    opt_res = bchh.get_optimization_result(dist_fn, None, phi_test, phi_original, tmp_distances, N_CLOSEST_TOPICS)
    cost_matrix = pd.DataFrame(100, index=phi_test.columns, columns=phi_original.columns)
    for idx, (column_name, opt_res_single_column) in enumerate(opt_res.iteritems()):
        if idx < cost_matrix.shape[1]:
            cost_matrix.loc[column_name, phi_original.columns[idx]] = opt_res_single_column['fun']
    return cost_matrix

#======================================================= plot ===========================================================================

def plot_original_columns_count_different_distances(different_distances, n_original_columns_count, title=''):
    sns.set_style("darkgrid")
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,5))

    if dh.jaccard_dist in different_distances.keys():    
        threshold_and_original_columns_count = different_distances[dh.jaccard_dist][0]
        x,y = zip(*threshold_and_original_columns_count)
        ax1.plot(x, y, 'ro-', label='jaccard')

    if dh.cos_dist in different_distances.keys():    
        threshold_and_original_columns_count = different_distances[dh.cos_dist][0]
        x,y = zip(*threshold_and_original_columns_count)
        ax1.plot(x, y, 'bo-', label='cos')

    if dh.hellinger_dist in different_distances.keys():    
        threshold_and_original_columns_count = different_distances[dh.hellinger_dist][0]
        x,y = zip(*threshold_and_original_columns_count)
        ax1.plot(x, y, 'yo-', label='hellinger')

    ax1.plot([0, 1], [n_original_columns_count, n_original_columns_count], linewidth=2, color='g', linestyle='--')

    ax1.set_xlabel('thresholds')
    ax1.set_ylabel('original shape count')
    ax1.legend()

    if dh.kl_dist in different_distances.keys():
        threshold_and_original_columns_count = different_distances[dh.kl_dist][0]
        x,y = zip(*threshold_and_original_columns_count)
        ax2.plot(x, y, 'bo-', label='kl')

    if dh.kl_sym_dist in different_distances.keys():
        threshold_and_original_columns_count = different_distances[dh.kl_sym_dist][0]
        x,y = zip(*threshold_and_original_columns_count)
        ax2.plot(x, y, 'mo-', label='kl sym')

    ax2.plot([0, 1], [n_original_columns_count, n_original_columns_count], linewidth=2, color='g', linestyle='--')

    ax2.set_xlabel('thresholds')
    ax2.set_ylabel('original shape count')
    ax2.legend()
    if title != '':
        fig.suptitle(title, fontsize=14, fontweight='bold')
 
from os import path, mkdir
from sys import stdout
from IPython.display import display
import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import distances_helper as dh 

def dynamic_print(data):
    stdout.write("\r\x1b[K"+data.__str__())
    stdout.flush()

def non_zero_ratio(phi, plot=False):
    eps = 1e-15
    words_count, topics_count = phi.shape
    # total
    non_zero_count = np.where(phi < eps)[0].shape[0]
    non_zero_ratio = 1.0 * non_zero_count / (words_count * topics_count)    
    # each column
    non_zero_count_by_columns = [np.sum(topic < eps) for topic_name, topic in phi.iteritems()]
    non_zero_ratio_by_columns = [1.0 * val / words_count for val in non_zero_count_by_columns]                                 
    if plot:
        sns.distplot(non_zero_ratio_by_columns)
    return non_zero_ratio, non_zero_ratio_by_columns
def words_probs_larger_th_count(phi, th):
    return np.sum(np.sum(phi > th)) * 1.0 / phi.shape[1]
def words_probs_smaller_th_count(phi, th):
    return np.sum(np.sum(phi < th)) * 1.0 / phi.shape[1]
def print_models_comparasion(phi1, phi2, theta1=None, theta2=None):
    models_compare_matrix = pd.DataFrame(0, columns=['phi_1', 'phi_2'], index=[])
    models_compare_matrix.loc['num words', :] = [phi1.shape[0], phi2.shape[0]]
    models_compare_matrix.loc['num topics', :] = [phi1.shape[1], phi2.shape[1]]
    models_compare_matrix.loc['non zero ratio', :] = [non_zero_ratio(phi1)[0], non_zero_ratio(phi2)[0]]
    models_compare_matrix.loc['|phi_ij == 0| / n_topics', :] = [words_probs_smaller_th_count(phi1, 1e-15), 
                                                                words_probs_smaller_th_count(phi2, 1e-15)]
    
    models_compare_matrix.loc['|phi_ij > 0.2| / n_topics', :] = [words_probs_larger_th_count(phi1, 0.2), 
                                                                 words_probs_larger_th_count(phi2, 0.2)]
    models_compare_matrix.loc['|phi_ij > 0.1| / n_topics', :] = [words_probs_larger_th_count(phi1, 0.1), 
                                                                 words_probs_larger_th_count(phi2, 0.1)]
    models_compare_matrix.loc['|phi_ij > 0.01| / n_topics', :] = [words_probs_larger_th_count(phi1, 0.01), 
                                                                  words_probs_larger_th_count(phi2, 0.01)]
    models_compare_matrix.loc['|phi_ij > 0.05| / n_topics', :] = [words_probs_larger_th_count(phi1, 0.05), 
                                                                  words_probs_larger_th_count(phi2, 0.05)]
    models_compare_matrix.loc['|phi_ij > 0.001| / n_topics', :] = [words_probs_larger_th_count(phi1, 0.001), 
                                                                  words_probs_larger_th_count(phi2, 0.001)]
    
    pd.set_option('precision', 2)

    display(models_compare_matrix)
    return models_compare_matrix
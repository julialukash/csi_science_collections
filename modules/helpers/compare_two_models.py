from os import path, mkdir
from sys import stdout
from IPython.display import display
import datetime

import artm
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

def print_models_comparasion(phi1, phi2, phi_nwt1=[], phi_nwt2=[], theta1=[], theta2=[]):
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
    
    topic_kernel_average_size1, topic_kernel_average_purity1, topic_kernel_average_contrast1 = None, None, None
    topic_kernel_average_size2, topic_kernel_average_purity2, topic_kernel_average_contrast2 = None, None, None
    if len(phi_nwt1) and phi_nwt1.shape[1] != 0:
        topic_kernel_average_size1, topic_kernel_average_purity1, topic_kernel_average_contrast1 = \
        get_kernel_scores(phi1, phi_nwt1)
    elif len(phi_nwt1):
        topic_kernel_average_size1, topic_kernel_average_purity1, topic_kernel_average_contrast1 = 23.68, 0.629848021471, 0.776075638995
    if len(phi_nwt2):
        topic_kernel_average_size2, topic_kernel_average_purity2, topic_kernel_average_contrast2 = \
        get_kernel_scores(phi2, phi_nwt2)    
    if topic_kernel_average_size1 is not None or topic_kernel_average_size2 is not None:
        models_compare_matrix.loc['topic_kernel_average_size', :] = [topic_kernel_average_size1, topic_kernel_average_size2]
        models_compare_matrix.loc['topic_kernel_average_purity', :] = [topic_kernel_average_purity1, topic_kernel_average_purity2]
        models_compare_matrix.loc['topic_kernel_average_contrast', :] = [topic_kernel_average_contrast1, topic_kernel_average_contrast2]
    
    pd.set_option('precision', 2)

    display(models_compare_matrix)
    return models_compare_matrix

def get_bigartm_scores(create_model_fn, n_iteration, phi, debug=False):
    n_topics = phi.shape[1]
    tm_model = create_model_fn(n_iteration, fit=False, n_topics=n_topics)
    
    (_, phi_ref) = tm_model.master.attach_model(model=tm_model.model_pwt)

    if debug:
        for model_description in tm_model.info.model:
            print model_description
    np.copyto(phi_ref, phi.values) 

    # create batch_vectorizer
    vocabulary = {idx: word for idx, word in enumerate(phi.index)}
    X_values = np.array(np.zeros((len(vocabulary), 1)), dtype=np.float)
    test_batch_vectorizer = artm.BatchVectorizer(data_format='bow_n_wd',
                                                 n_wd=X_values,
                                                 vocabulary=vocabulary)
    tm_model.fit_offline(batch_vectorizer=test_batch_vectorizer, num_collection_passes=1)
    new_phi = tm_model.get_phi()
    not_equal_elements_count = np.sum(np.sum(np.abs(new_phi - phi) > 1e-3))
    if not_equal_elements_count > 0:
        print('New phi matrix is not equal to the old one, not_equal_elements_count = {}'.format(not_equal_elements_count)) 
    # only for not complex models
    return tm_model.score_tracker['perplexity_score'].last_value, tm_model.score_tracker['ss_phi_score'].last_value, \
           tm_model.score_tracker['topic_kernel_score'].last_average_size,\
           tm_model.score_tracker['topic_kernel_score'].last_average_purity, \
           tm_model.score_tracker['topic_kernel_score'].last_average_contrast
            
def get_kernel_scores(p_wt, n_wt, debug=False):
    delta_th = 0.25
    kernel_size = {topic: 0 for topic in p_wt.columns}
    kernel_purity = {topic: 0 for topic in p_wt.columns}
    kernel_contrast = {topic: 0 for topic in p_wt.columns}
    n_t = np.sum(n_wt)    
    for word in p_wt.index:
        p_w = np.sum(p_wt.loc[word, :] * n_t)
        if p_w:
            for topic in p_wt.columns:
                p_wt_single = p_wt.loc[word, topic]
                p_tw = p_wt_single * n_t[topic] / p_w 
                if p_tw >= 0.25:
                    kernel_size[topic] += 1
                    kernel_purity[topic] += p_wt_single
                    kernel_contrast[topic] += p_tw
    for topic in kernel_contrast.keys():
        if kernel_size[topic]:
            kernel_contrast[topic] /= kernel_size[topic]
    topic_kernel_average_size = np.mean(kernel_size.values())
    topic_kernel_average_purity = np.mean(kernel_purity.values())
    topic_kernel_average_contrast = np.mean(kernel_contrast.values()) 
    return topic_kernel_average_size, topic_kernel_average_purity, topic_kernel_average_contrast
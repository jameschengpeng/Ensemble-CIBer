import numpy as np
import pandas as pd
import copy
import operator
import tool_box as tb
from tool_box import empty_intersection_correction
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import resample
from scipy import stats
from mdlp.discretization import MDLP
from sklearn.metrics import normalized_mutual_info_score
from math import log2
from pyitlib import discrete_random_variable as drv
from scipy.spatial import distance
from numba import jit
import timeit

# get the normalized mutual information between two variables
# i j are column indices, x, y are two column values
def single_norm_mutual_info(i,j,x,y):
    return i, j, normalized_mutual_info_score(x, y,average_method='arithmetic')

def get_norm_mutual_info(var_4_cluster):
    n_var = var_4_cluster.shape[1]
    result = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(i, n_var):
            if i == j:
                result[i][j] = 1
            else:
                result[i][j] = result[j][i] = normalized_mutual_info_score(var_4_cluster[:,i], 
                                                                           var_4_cluster[:,j], average_method='arithmetic')
    return result

def get_CMI(var_4_cluster, y):
    n_var = var_4_cluster.shape[1]
    result = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(i, n_var):
            if i == j:
                result[i,j] = result[j,i] = 1
            else:
                var1 = list(var_4_cluster[:,i].flatten())
                var2 = list(var_4_cluster[:,j].flatten())
                result[i,j] = result[j,i] = drv.information_mutual_conditional(var1, var2, y)
    return result

def get_JS_matrix(var_4_cluster):
    n_var = var_4_cluster.shape[1]
    result = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(i, n_var):
            if i == j:
                result[i,j] = result[j,i] = 0
            else:
                var1 = var_4_cluster[:,i]
                var2 = var_4_cluster[:,j]
                const_i = np.all(var1 == var1[0])
                const_j = np.all(var2 == var2[0])
                if const_i or const_j:
                    result[i,j] = result[j,i] = 1
                else:
                    result[i,j] = result[j,i] = distance.jensenshannon(var1, var2)
    return result


# apply joint encoding to the categorical variables, do not cluster them in advance
# discretize continuous variables, then cluster all variables together
# discrete_feature_val: {idx_of_discrete_feature_val: number of distinct feature values of this feature}
# allocation_book: {idx_of_cont_feature: number of bins to discretize}
class clustered_comonotonic:
    # discrete is treated as cont
    def __init__(self, x_train, y_train, discrete_feature_val, cont_col, categorical, 
                 min_corr, corrtype='pearson', discrete_method = 'auto', allocation_book = None):
        self.x_train = x_train
        self.y_train = y_train
        self.discrete_feature_val = discrete_feature_val
        self.cont_col = cont_col
        self.min_corr = min_corr
        self.discrete_method = discrete_method
        self.allocation_book = allocation_book
        self.corrtype = corrtype
        if len(cont_col) == 0:
            self.cont_feature_val = dict()

    def discretize(self):
        if self.discrete_method != 'mdlp':
            cont_feature_val = dict()
            for i in range(self.x_train.shape[1]):
                if i in self.cont_col:
                    if self.discrete_method == 'auto':
                        cont_feature_val[i] = 8
                    else:
                        cont_feature_val[i] = self.allocation_book[i]
            x_transpose = self.x_train.T.copy()
            discrete_x = []
            bin_info = {k:None for k in self.cont_col}
            for i, feature in enumerate(x_transpose):
                if i in self.cont_col:
                    if self.discrete_method == 'auto':
                        discretized, bins = tb.auto_discretize(feature)
                        discrete_x.append(discretized)
                        bin_info[i] = bins.copy()
                    else:
                        discretized, bins = tb.custom_discretize(feature, self.allocation_book[i])
                        discrete_x.append(discretized)
                        bin_info[i] = bins.copy()
                else:
                    discrete_x.append(feature)
            self.x_train = np.array(discrete_x).T.astype(int)
            self.bin_info = bin_info  
        else:
            transformer = MDLP()
            training_copy = self.x_train[:,self.cont_col].copy()
            training_copy = transformer.fit_transform(training_copy, self.y_train)
            self.transformer = transformer
            discrete_x = []
            x_transpose = self.x_train.T.copy()
            bin_info = dict()
            self.mixed_discrete = False
            index_cont = 0 # index within the continuous features list
            for i, feature in enumerate(x_transpose):
                if i in self.cont_col:
                    # if just one value after discretization use auto discretize instead
                    if min(training_copy.T[index_cont]) == max(training_copy.T[index_cont]):
                        self.mixed_discrete = True
                        discretized, bins = tb.auto_discretize(feature)
                        discrete_x.append(discretized)
                        bin_info[i] = bins.copy()
                    else:
                        discrete_x.append(training_copy.T[index_cont])
                    index_cont += 1
                else:
                    discrete_x.append(x_transpose[i])
            self.x_train = np.array(discrete_x).T.astype(int)  
            self.bin_info = bin_info
        # feature_val
        # cont_feature_val: {idx_of_cont_feature: number of distinct values after discretization}
        cont_feature_val = {}
        for i in range(len(self.x_train.T)):
            if i in self.cont_col:
                cont_feature_val[i] = max(self.x_train.T[i])+1
        self.cont_feature_val = cont_feature_val
        # feature_val: {idx of every feature: number of distinct feature values}
        if self.discrete_feature_val != None:
            # the number of distinct values for each feature (including cont after discretization)
            self.feature_val = tb.merge_dict(cont_feature_val, self.discrete_feature_val)
        else: # all continuous 
            self.feature_val = cont_feature_val.copy()

    # in case there is no continuous feature, call this function to obtain the feature value
    def construct_feature_val(self):
        if self.discrete_feature_val != None:
            self.feature_val = tb.merge_dict(self.cont_feature_val, self.discrete_feature_val)
        else:
            self.feature_val = self.cont_feature_val.copy()

    def clustering(self):
        if self.min_corr == 0:
            self.cluster_book = [[i for i in self.cont_col]]
        elif self.min_corr == 1:
            self.cluster_book = [[i] for i in self.cont_col]
        else:
            # now the data has been discretized and joint encoded, cluster together
            var_4_cluster = self.x_train.copy()
            idx_4_cluster = [i for i in range(self.x_train.shape[1])]
            if self.corrtype == 'pearson':
                corr_matrix = np.corrcoef(var_4_cluster.T)
            elif self.corrtype == 'spearman':
                corr_matrix = stats.spearmanr(var_4_cluster)[0]
            elif self.corrtype == 'kendall':
                temp_df = pd.DataFrame(data = var_4_cluster)
                corr_matrix = temp_df.corr(method='kendall').to_numpy()
            elif self.corrtype == 'mutual_info':
                corr_matrix = get_norm_mutual_info(var_4_cluster)
            elif self.corrtype == 'cmi':
                y = list(self.y_train.flatten())
                corr_matrix = get_CMI(var_4_cluster, y)
            elif self.corrtype == 'js_divergence': # measure by Jensen-Shannon divergence
                corr_matrix = 1 - get_JS_matrix(var_4_cluster)
            if self.x_train.shape[1] == 2:
                corr_matrix = np.array([[1,corr_matrix],[corr_matrix,1]])
            
            abs_corr = np.absolute(corr_matrix)
            distance_matrix = 1 - abs_corr

            clusterer = AgglomerativeClustering(affinity='precomputed', linkage='average', 
                                                distance_threshold=1-self.min_corr, n_clusters=None)
            clusterer.fit(distance_matrix)
            adjusted_cluster_dict = dict()
            # think about simplification here, delete idx_4_cluster
            for i,c in enumerate(clusterer.labels_):
                if c not in adjusted_cluster_dict.keys():
                    adjusted_cluster_dict[c] = list()
                adjusted_cluster_dict[c].append(idx_4_cluster[i])
            adjusted_cluster_book = list()
            for k in adjusted_cluster_dict.keys():
                adjusted_cluster_book.append(adjusted_cluster_dict[k].copy())
            self.cluster_book = adjusted_cluster_book

    def get_prior_prob(self):
        # get the prior probability and indices of instances for different classes
        # prior_prob: {class: the proportion of data in this class}
        prior_prob = dict() # key is class, value is the prior probability of this class
        class_idx = dict() # key is class, value is a list containing the indices of instances for this class
        for value in np.unique(self.y_train):
            class_idx[value] = np.squeeze(np.where(self.y_train == value)).tolist()
        prior_prob = {k:len(v)/len(self.y_train) for k,v in class_idx.items()}
        self.prior_prob = prior_prob
        self.class_idx = class_idx

    # {feature: {class: {feature_value: probability} } }
    def get_cond_prob(self, feature_list):
        cond_prob = dict()
        for f in feature_list:
            feature_dict = dict()
            for c in self.class_idx.keys():
                # indices is a list containing the indices of data points in this class
                indices = self.class_idx[c]
                sample_val = self.x_train[indices, f]
                val_counting = np.unique(sample_val, return_counts = True)
                class_dict = dict(zip(val_counting[0], val_counting[1]))
                #laplacian correction
                all_fv = [i for i in range(self.feature_val[f])]
                for fv in all_fv:
                    if fv not in list(class_dict.keys()):
                        class_dict[fv] = 1
                    else:
                        class_dict[fv] += 1
                if 0 in list(class_dict.values()):
                    for fv in class_dict.keys():
                        class_dict[fv] += 1
                summation = sum(class_dict.values())
                class_dict = {k:v/summation for k,v in class_dict.items()}
                feature_dict[c] = class_dict
            cond_prob[f] = feature_dict
        return cond_prob                

    # consider deleting this function
    def complete_posterior_prob(self):
        como_feature_list = [i for i in range(self.x_train.shape[1])]
        self.como_post_prob = self.get_cond_prob(como_feature_list)

    def get_prob_interval(self, class_dict, feature_value):
        # class_dict :: {feature_value: posterior_prob}
        if feature_value == 0:
            return [0, class_dict[feature_value]]
        else:
            inf = sup = 0
            for i in class_dict.keys():
                if i < feature_value:
                    inf += class_dict[i]
            sup = inf + class_dict[feature_value]
            return [inf, sup]

    def get_como_prob_interval(self):
        # get the interval for posterior prob
        # { feature_idx: { class: {feature_value: [inf, sup] } } }
        prob_interval_collection = dict()
        for cluster in self.cluster_book:
            if len(cluster) > 1:
                como_var = self.x_train[:, cluster]
                corr_matrix = stats.spearmanr(como_var)[0]
                if len(cluster) == 2:
                    corr_matrix = np.array([[1, corr_matrix],[corr_matrix, 1]])
                corr_sum = [sum([abs(j) for j in corr_matrix[i]]) for i in range(len(corr_matrix))]
                base_feature_idx = corr_sum.index(max(corr_sum))
                base_feature = cluster[base_feature_idx]
                for f in cluster:
                    feature_dict = dict()
                    for c in self.como_post_prob[f].keys():
                        class_dict = dict()
                        for fv in self.como_post_prob[f][c].keys():
                            interval = self.get_prob_interval(self.como_post_prob[f][c], fv)
                            if f != base_feature:
                                feature_como_idx = cluster.index(f)
                                if corr_matrix[base_feature_idx][feature_como_idx] < 0:
                                    interval = [1-interval[1], 1-interval[0]]
                            class_dict[fv] = interval
                        feature_dict[c] = class_dict
                    prob_interval_collection[f] = feature_dict            
            else: #only one feature in this cluster
                f = cluster[0]
                feature_dict = dict()
                for c in self.como_post_prob[f].keys():
                    class_dict = dict()
                    for fv in self.como_post_prob[f][c].keys():
                        class_dict[fv] = self.get_prob_interval(self.como_post_prob[f][c], fv)
                    feature_dict[c] = class_dict
                prob_interval_collection[f] = feature_dict
        self.como_prob_interval = prob_interval_collection
    
    def run(self):
        start = timeit.default_timer()
        if len(self.cont_col) != 0:
            # try discretize first
            self.clustering()
            time1 = timeit.default_timer()
            self.discretize()
            time2 = timeit.default_timer()
        if len(self.cont_col) == 0:
            self.construct_feature_val()
        self.get_prior_prob()
        self.complete_posterior_prob()
        self.get_como_prob_interval()
        time3 = timeit.default_timer()
        print("Training (except mdlp) time: " + str(time3 - time2 + time1 - start))
    
    def interval_intersection(self, intervals): # intervals is a list of list
        infimum = max([interval[0] for interval in intervals])
        supremum = min([interval[1] for interval in intervals])
        if supremum > infimum:
            return (supremum-infimum)
        else:
            #return 0.001
            return 0
    
    def get_prob_dist_single(self, x):
        start = timeit.default_timer()
        if len(self.cont_col) != 0:
            # change x to categorical
            cate_x = list()
            if self.discrete_method != 'mdlp':
                for i,f in enumerate(x):
                    if i in self.cont_col:
                        cate_x.append(np.digitize(f,self.bin_info[i]))
                    else:
                        cate_x.append(f)
            else:
                x_copy = x.copy()
                x_copy = np.array(x_copy).reshape(1,-1)
                x_copy_cont = x_copy[:,self.cont_col]
                x_copy_cont = self.transformer.transform(x_copy_cont)
                index_cont = 0
                for i,f in enumerate(x):
                    if self.mixed_discrete == False:
                        if i in self.cont_col:
                            cate_x.append(x_copy_cont[0][index_cont])
                            index_cont += 1
                        else:
                            cate_x.append(f)
                    else:
                        if (i in self.cont_col) and (i not in self.bin_info.keys()):
                            cate_x.append(x_copy_cont[0][index_cont])
                            index_cont += 1
                        elif (i in self.cont_col) and (i in self.bin_info.keys()):
                            cate_x.append(np.digitize(f,self.bin_info[i]))
                            index_cont += 1
                        else:
                            cate_x.append(f)
        else:
            cate_x = x.copy()
        end = timeit.default_timer()
        # get the probability distribution of one instance
        prob_distribution = self.prior_prob.copy() # initialize with prior probability
        backup_prob_dist = prob_distribution.copy()
        for c in prob_distribution.keys():
            for cluster in self.cluster_book:
                if len(cluster) == 1:
                    f_idx = cluster[0]
                    fv = cate_x[f_idx]
                    try:
                        interval = self.como_prob_interval[f_idx][c][fv]
                        prob_distribution[c] *= interval[1] - interval[0]
                    except:
                        prob_distribution[c] *= 0.01
                else:
                    intervals = []
                    for f_idx in cluster:
                        fv = cate_x[f_idx]
                        if fv in self.como_prob_interval[f_idx][c].keys():
                            intervals.append(self.como_prob_interval[f_idx][c][fv])
                        else:
                            max_key = max(self.como_prob_interval[f_idx][c].keys())
                            min_key = min(self.como_prob_interval[f_idx][c].keys())
                            if fv > max_key:
                                intervals.append(self.como_prob_interval[f_idx][c][max_key])
                            else:
                                intervals.append(self.como_prob_interval[f_idx][c][min_key])
                    # if the intersection is empty, then find pairwise intersections, and treat pairs as independent
                    intersection_length = self.interval_intersection(intervals)
                    if intersection_length != 0:
                        prob_distribution[c] *= intersection_length
                    else:
                        prob_distribution[c] *= 0.001
                        #prob_distribution[c] *= empty_intersection_correction(intervals)
        checker = sum(list(prob_distribution.values()))
        if checker == 0:
            summation = sum(list(backup_prob_dist.values()))
            final_distribution = {}
            for k in backup_prob_dist.keys():
                final_distribution[k] = backup_prob_dist[k]/summation 
            return final_distribution, end - start
        else:
            summation = sum(list(prob_distribution.values()))
            final_distribution = {}
            for k in prob_distribution.keys():
                final_distribution[k] = prob_distribution[k]/summation
            return final_distribution, end - start
        
    def predict_proba(self, x_test):
        y_predict = list()
        for x in x_test:
            distribution,_ = self.get_prob_dist_single(x)
            y_predict.append(list(distribution.values()))
        return np.array(y_predict)
    
    def predict(self, x_test):
        start = timeit.default_timer()
        y_predict = list()
        discretization_time = list()
        for x in x_test:
            prob_dist,t = self.get_prob_dist_single(x)
            discretization_time.append(t)
            predicted_class = max(prob_dist.items(), key=operator.itemgetter(1))[0]
            y_predict.append(predicted_class)
        end = timeit.default_timer()
        print('Testing time: ' + str(end-start-sum(discretization_time)))
        return y_predict
    
    def print_cluster(self):
        return self.cluster_book
import numpy as np
import pandas as pd
import copy
import comonotonic as cm
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import random
import math
from math import log2
import operator
from scipy.stats import norm
import statistics
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# given a list containing text values in a column
# encode to 0, 1, 2...
def encoding(column_val):
    encountered = {} # {text value: encoder}
    encoded = []
    encoder = 0
    for val in column_val:
        if val not in encountered.keys():
            encountered[val] = encoder
            encoder += 1
        encoded.append(encountered[val])
    return encoded

def encode_df(df, encoded_columns):
    df_copy = df.copy()
    for col in encoded_columns:
        encoded = encoding(df.iloc[:,col])
        if col == df.shape[1] - 1: # Y needs to be encoded
            df_copy['Y'] = encoded
        else:
            df_copy['X'+str(col)] = encoded
    return df_copy

def normalize_df_col(df, cont_columns):
    for col in cont_columns:
        colname = 'X' + str(col)
        col_val = df[[colname]].values
        col_val = list(col_val.ravel())
        mean = np.mean(col_val)
        std = np.std(col_val)
        col_scaled = list(map(lambda x:(x-mean)/std, col_val))
        df[colname] = col_scaled

# split the dataframe, discretize the training set and get the bins, discretize the test set based on the bins
def df_split_discretize(df, random_state, qcut, allocation_book):
    X = df.iloc[:,0:-1].copy()
    Y = df.iloc[:,-1].copy()
    for col_idx in allocation_book.keys():
        if qcut == False:    
            discretized_col = pd.cut(X.iloc[:,col_idx],allocation_book[col_idx], labels=[i for i in range(allocation_book[col_idx])])
            discretized_col = discretized_col.astype('int32')
        else:
            try:
                discretized_col = pd.qcut(X.iloc[:,col_idx],allocation_book[col_idx], 
                                        labels=[i for i in range(allocation_book[col_idx])])
            except:
                discretized_col = pd.cut(X.iloc[:,col_idx],allocation_book[col_idx], labels=[i for i in range(allocation_book[col_idx])])
            discretized_col = discretized_col.astype('int32')
        X['X'+str(col_idx)] = discretized_col
    X_train_df, X_test_df, Y_train_df, Y_test_df = train_test_split(X,Y,test_size=0.2,random_state=random_state)
    return X_train_df.to_numpy(), X_test_df.to_numpy(), Y_train_df.to_numpy(), Y_test_df.to_numpy()

def merge_dict(d1, d2):
    d = {}
    for k1 in d1.keys():
        d[k1] = d1[k1]
    for k2 in d2.keys():
        d[k2] = d2[k2]
    return d

def get_accuracy(y_predict, y_test):
    t = 0
    for i in range(len(y_test)):
        if y_predict[i] == y_test[i]:
            t += 1        
    return t/len(y_test)

######## for clustering by correlation coefficient matrix
# c1, c2 are two lists, compute the distance between them
def get_distance(distance_matrix, c1, c2):
    min_dist = -1
    for f1 in c1:
        for f2 in c2:
            dist = distance_matrix[f1][f2]
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
    return min_dist

# update cluster_book
def update_cluster_book(distance_matrix, cluster_book):
    min_dist = -1
    merger1 = None
    merger2 = None
    for c1 in cluster_book:
        for c2 in cluster_book:
            if c1 != c2:
                dist = get_distance(distance_matrix, c1, c2)
                if min_dist == -1 or min_dist > dist:
                    merger1 = c1.copy()
                    merger2 = c2.copy()
                    min_dist = dist
    if merger1 == None or merger2 == None:
        return cluster_book, 1
    cluster_book.remove(merger1)
    cluster_book.remove(merger2)
    cluster_book.append(merger1+merger2)
    return cluster_book, min_dist

# stopping condition: the current minimum distance among clusters larger than max_distance
def cluster_agnes(distance_matrix, max_distance):
    cluster_book = [[i] for i in range(len(distance_matrix))] # list of list, store the current clusters
    cluster_book_copy = cluster_book.copy() # in case clustering is unnecessary
    cluster_book, min_dist = update_cluster_book(distance_matrix, cluster_book) 
    cluster_necessary = False # if clustering is unnecessary, just use cluster_book_copy
    while min_dist <= max_distance:
        cluster_necessary = True
        cluster_book, min_dist = update_cluster_book(distance_matrix, cluster_book)
    if cluster_necessary == False:
        return cluster_book_copy
    else:
        return cluster_book
    

# use weighted average of naive bayes and pure comonotonic
def weighted_avg(nb_dist, p_como_dist, w_nb):
    prob_dist = {c: (nb_dist[c]*w_nb+p_como_dist[c]*(1-w_nb)) for c in nb_dist.keys()}
    return max(prob_dist.items(), key=operator.itemgetter(1))[0]

# scale the probability density to magnitude 10^(-1)
def scaler(prob_density):
    i = 0
    while prob_density < 0.1:
        prob_density *= 10
        i += 1
    return i, prob_density

# pass in a numpy 1d array of samples of a continuous variable
def auto_discretize(feature):
    mean = np.mean(feature)
    std = np.std(feature)
    bins = np.array([mean-3*std, mean-2*std, mean-std, mean, mean+std, mean+2*std, mean+3*std])
    discretized = np.digitize(feature, bins)
    return discretized, bins

def custom_discretize(feature, num_classes):
    sup = max(feature)
    inf = min(feature)
    stride = (sup - inf)/num_classes
    bins = [inf+stride*(i+1) for i in range(num_classes-1)]
    discretized = np.digitize(feature, bins)
    return discretized, bins


def outlier_removal(df, col = None):
    if col == None:
        X = df.iloc[:,:-1].to_numpy().T
    else:
        X = df.iloc[:,col].to_numpy().T
    cov = np.cov(X)
    inv_cov = np.linalg.inv(cov)
    mean = np.mean(X, axis = 1)
    subtract_mean = np.array([[X[i,j]-mean[i] for j in range(X.shape[1])] for i in range(X.shape[0])]).T
    mahalanobis_distance = np.array([math.sqrt(subtract_mean[i].dot(inv_cov).dot(subtract_mean[i].T))
                                     for i in range(subtract_mean.shape[0])])
    mahalanobis_mean = np.mean(mahalanobis_distance)
    mahalanobis_std = np.std(mahalanobis_distance)
    threshold = mahalanobis_mean + 3*mahalanobis_std
    outliers = []
    for i in range(len(mahalanobis_distance)):
        if mahalanobis_distance[i] > threshold:
            outliers.append(i)
    reduced_df = df.drop(df.index[outliers])
    return reduced_df


# return a df of distinct value combinations in descending order of frequencies
# pass in a df with categorical features
# the df only contains not fully encoded features 
def get_frequencies(df_unencode):
    cols = list(df_unencode.columns)
    frequency_df = df_unencode.groupby(cols).size().reset_index().rename(columns={0:'count'}).copy()
    frequency_df = frequency_df.sort_values(by=['count'], ascending = False).reset_index(drop=True)
    frequency_df = frequency_df.drop(['count'], axis = 1)
    return frequency_df

# count the frequencies of each value in dictionary
def dict_value_count(dictionary):
    val_count = {val:0 for val in dictionary.values()}
    for k in dictionary.keys():
        val = dictionary[k]
        val_count[val] += 1
    return val_count

# if there is a unique none in a dictionary's value
# extract the key and the maximum of other values
def find_unique_none(dictionary):
    max_val = -1
    key = None
    for k in dictionary.keys():
        if dictionary[k] == None:
            key = k
        else:
            if dictionary[k] > max_val:
                max_val = dictionary[k]
    return max_val, key

def joint_encode(df, cluster):
    # the existing columns
    col_exist = [df.columns[i] for i in cluster]
    # the dataframe with feature that are not fully encoded
    df_unencode = df[col_exist].copy()
    # store the encoding information
    encode_ref = {col:{val:None for val in df_unencode[col].unique()} for col in col_exist}
    # track those fully encoded columns
    fully_encoded = {col:False for col in col_exist}

    
    while df_unencode.shape[1] > 0:
        frequency_df = get_frequencies(df_unencode)
        # if any feature in the following for-loop is fully encoded
        # set breaker = True, and break the for-loop
        # we need a new df_unencode
        breaker = False
        ######################
        # in case that there is no eligible row in frequency_df
        no_eligible_row = True
        # to record the row index with maximum number of unencoded values
        # and the maximum number of unencoded values
        most_unencoded_row = -1
        most_unencoded = -1

        for i in range(frequency_df.shape[0]):
            # check if this row is eligible for encoding
            # value of every existing feature should be unencoded yet
            eligible = True
            new_encode = dict() # {feature: (value, encoded num)}
            unencoded_num = frequency_df.shape[1]
            for col in frequency_df.columns:
                val = frequency_df.iloc[i,][col]
                # this feature value has been encoded
                if encode_ref[col][val] != None:
                    unencoded_num -= 1
                else:
                    if all([v == None for v in encode_ref[col].values()]):
                        encode_num = 0
                    else:
                        encode_num = max([v for v in encode_ref[col].values() if v != None]) + 1
                    new_encode[col] = tuple((val, encode_num))
            if unencoded_num < frequency_df.shape[1]:
                eligible = False
            if unencoded_num > most_unencoded:
                most_unencoded = unencoded_num
                most_unencoded_row = i
            # this row can be used
            if eligible:
                no_eligible_row = False
                for col in new_encode.keys():
                    val = new_encode[col][0]
                    encode_num = new_encode[col][1]
                    encode_ref[col][val] = encode_num
                    # check if this feature is fully encoded
                    if None not in encode_ref[col].values():
                        fully_encoded[col] = True
                        breaker = True
                    # if this feature has only 1 unencoded value, encode directly
                    val_count = dict_value_count(encode_ref[col])
                    if val_count[None] == 1:
                        max_val, key = find_unique_none(encode_ref[col])
                        val = key
                        encode_num = max_val + 1
                        encode_ref[col][val] = encode_num
                        fully_encoded[col] = True
                        breaker = True

            if breaker:
                # update the df_unencode by keeping only not fully encoded features
                new_df_unencode = pd.DataFrame()
                for col in fully_encoded.keys():
                    if not fully_encoded[col]:
                        new_df_unencode[col] = df_unencode[col]
                df_unencode = new_df_unencode.copy()
                del new_df_unencode
                break

        # there is no row that is eligible
        if no_eligible_row: # then encode the row with most unencoded values
            new_encode = dict() # {feature: (value, encoded num)}
            for col in frequency_df.columns:
                val = frequency_df.iloc[most_unencoded_row,][col]
                # this feature value has been encoded
                if encode_ref[col][val] == None:
                    if all([v == None for v in encode_ref[col].values()]):
                        encode_num = 0
                    else:
                        encode_num = max([v for v in encode_ref[col].values() if v != None]) + 1
                    new_encode[col] = tuple((val, encode_num))
            for col in new_encode.keys():
                val = new_encode[col][0]
                encode_num = new_encode[col][1]
                encode_ref[col][val] = encode_num
                if None not in encode_ref[col].values():
                    fully_encoded[col] = True
                # if this feature has only 1 unencoded value, encode directly
                val_count = dict_value_count(encode_ref[col])
                if val_count[None] == 1:
                    max_val, key = find_unique_none(encode_ref[col])
                    val = key
                    encode_num = max_val + 1
                    encode_ref[col][val] = encode_num
                    fully_encoded[col] = True

            new_df_unencode = pd.DataFrame()
            for col in fully_encoded.keys():
                if not fully_encoded[col]:
                    new_df_unencode[col] = df_unencode[col]
            df_unencode = new_df_unencode.copy()
            del new_df_unencode

    for col in encode_ref.keys():
        df[col].replace(encode_ref[col], inplace = True)
    return df, encode_ref






# sort a categorical feature by the frequency of each value, largest -> 0, 2nd largest -> 1, etc.
def simple_encode(df, cate_col):
    colnames = [df.columns[i] for i in cate_col]
    for col in colnames:
        occurence_dict = df[col].value_counts(ascending = False)
        encode_dict = dict()
        for i, k in enumerate(occurence_dict.keys()):
            encode_dict[k] = i
        df[col].replace(encode_dict, inplace = True)
    return df

# compute the cross entropy error
def cross_entropy(x_proba, y_val):
    loss = 0
    for i in range(x_proba.shape[0]):
        prob_dist = x_proba[i]
        y = y_val[i]
        if prob_dist[y] != 0:
            loss += -log2(prob_dist[y])
        else:
            loss += -log2(10**(-3))
    return loss/x_proba.shape[0]

def rmse(y, y_pred, y_prob_dist):
    mse = 0
    for i in range(y_prob_dist.shape[0]):
        real_prob_dist = np.array([0 for j in range(y_prob_dist.shape[1])])
        real_prob_dist[y[i]] = 1
        pred_prob_dist = y_prob_dist[i]
        mse += sum([j**2 for j in real_prob_dist-pred_prob_dist])
    mse = mse/len(y)
    return math.sqrt(mse)

def interval_intersection(interval1, interval2):
    inf = max(interval1[0], interval2[0]) 
    sup = min(interval1[1], interval2[1])
    if inf < sup:
        return sup - inf
    else:
        return 0

# make correction when the intersection is empty; pass in a list of list where the sublists are intervals
# iteratively find pairs of intervals having the maximum intersection
def empty_intersection_correction(intervals):
    recorder = dict()
    como_idx = list()
    corrected_val = 1
    for i in range(len(intervals)):
        for j in range(i+1,len(intervals)):
            recorder[(i,j)] = interval_intersection(intervals[i], intervals[j])
    while bool(recorder): # recorder is not empty
        if max(recorder.values()) > 0:
            max_key = max(recorder.items(), key=operator.itemgetter(1))[0]
            corrected_val *= recorder[max_key]
            como_idx.append(max_key[0])
            como_idx.append(max_key[1])
            all_keys = list(recorder.keys())
            for key in all_keys:
                if key[0] in max_key or key[1] in max_key:
                    del(recorder[key])
            del all_keys
    for i in range(len(intervals)):
        if i not in como_idx:
            corrected_val *= (intervals[i][1] - intervals[i][0])
    return corrected_val

# x_proba is a 2d array containing the predicted probability distribution over all classes for each instance
# y is the actual class
def informational_loss(x_proba, y):
    loss = 0
    for i in range(len(y)):
        loss += -log2(x_proba[i,y[i]])
    return loss/len(y)
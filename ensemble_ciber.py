import comonotonic as cm
import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp
import operator
from collections import Counter
import tool_box as tb
from scipy import stats


# please joint encode x_train beforehand
class ciber_forest:
    
    def __init__(self, discrete_feature_val, cont_col, categorical, min_corr, corrtype, discrete_method,
                 max_samples = 0.5, max_features = 0.7, n_estimators = 10, max_workers = mp.cpu_count()):
        self.n_estimators = n_estimators
        self.max_workers = max_workers
        self.discrete_feature_val = discrete_feature_val
        self.cont_col = cont_col
        self.categorical = categorical
        self.min_corr = min_corr
        self.corrtype = corrtype
        self.discrete_method = discrete_method
        self.max_samples = max_samples
        self.max_features = max_features

    # singleton classifier
    # sequence_num is for the ease of extracting results from pooling
    def single_ciber(self, x_train, y_train, sequence_num):
        if self.max_samples <= 1:
            n_samples = int(x_train.shape[0] * self.max_samples)
        else:
            n_samples = self.max_samples
        if self.max_features <= 1:
            n_features = int(x_train.shape[1] * self.max_features)
        else:
            n_features = self.max_features
        # draw the features WITHOUT replacement; draw the samples WITH replacement
        row_idx = np.random.choice(x_train.shape[0], n_samples, replace = True)
        col_idx = np.random.choice(x_train.shape[1], n_features, replace = False)
        col_idx.sort()
        # sample from x_train, and y_train
        x_sample = x_train[row_idx,:][:,col_idx].copy()
        y_sample = y_train[row_idx].copy()
        # update others based on the sampled columns
        sample_discrete_feature_val = dict()
        sample_cont_col = list()
        sample_categorical = list()
        for i,col in enumerate(col_idx):
            if col in self.cont_col:
                sample_cont_col.append(i)
            else:
                sample_discrete_feature_val[i] = self.discrete_feature_val[col]
                sample_categorical.append(i)
        ciber_clf = cm.clustered_comonotonic(x_sample, y_sample, sample_discrete_feature_val, 
                                            sample_cont_col, sample_categorical, self.min_corr, 
                                            corrtype = self.corrtype, discrete_method = self.discrete_method)
        ciber_clf.run()
        return ciber_clf, col_idx, sequence_num

    # do parallel computing
    def parallel_fitting(self, x_train, y_train):
        param_collection = list()
        for i in range(self.n_estimators):
            params = (x_train.copy(), y_train.copy(), i)
            param_collection.append(params)

        pool = Pool(processes = self.max_workers)
        self.clf_collection = pool.starmap(self.single_ciber, param_collection)
        pool.close()

    def ensemble_predict(self, x_test):
        # each row is the prediction from one classifier
        prediction_collection = list()
        for clf, col_idx, _ in self.clf_collection:
            x_test_selected_col = x_test[:,col_idx].copy()
            prediction = clf.predict(x_test_selected_col)
            prediction_collection.append(prediction.copy())
            del prediction
        # now each row is the predictions for one sample point
        prediction_collection = np.array(prediction_collection).T
        final_prediction = stats.mode(prediction_collection, axis=1)[0].flatten()
        return final_prediction

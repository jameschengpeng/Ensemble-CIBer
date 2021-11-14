# Ensemble-CIBer
The ensemble learning version of CIBer

When using the code, please remember to define a function like the one below.

```python
def run(x_train, x_test, y_train, discrete_feature_val, cont_col, categorical, min_corr, 	corrtype, discrete_method,
        max_samples = 0.5, max_features = 0.7, n_estimators = 10, max_workers = mp.cpu_count()):
    if __name__ == '__main__':
        ensemble_ciber = ec.ciber_forest(discrete_feature_val, cont_col, categorical, 
                                      min_corr, corrtype, discrete_method)
        ensemble_ciber.parallel_fitting(x_train, y_train)
        prediction = ensemble_ciber.ensemble_predict(x_test)
        print('exist')
        return prediction
```

In other words, you need to do data processing including: joint encoding, figuring out the `discrete_feature_val` `cont_col` `categorical` from the data (please note that `cont_col` and `categorical` form a PARTITION of all the features), resample the dataset (if needed), and do train test splitting. Then, you may use the function above to apply `ensemble_ciber` to your data.

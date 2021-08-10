#  ===========================================================================
#  @file:   low_variance.py
#  @brief:  Eliminate the low variance columns given a threshold 
#  @author: Johan Mendez
#  @date:   14/05/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


def variance_threshold_selector(data, threshold=0.5):
    data_aux = data.drop([["GESTION_COBRO","SALDADA"]], axis=1)
    selector = VarianceThreshold(threshold)
    selector.fit(data_aux)
    data_aux = data_aux[data_aux.columns[selector.get_support(indices=True)]]
    data = np.concatenate(data_aux,data[["GESTION_COBRO", "SALDADA"]] ,axis=1)
    return data


def Remove_Outliers(dataframe):

    iso = IsolationForest(#n_estimators=10,
                          #max_samples="auto",
                          contamination=0.1,
                          #max_features=1.0,
                          #bootstrap=False,
                          n_jobs=-1,
                          random_state=11,
                          verbose=0
                          )

    outliers = iso.fit_predict(dataframe)

    return dataframe[outliers != -1]

def remove_records_with_outliers(dataset, ignore_columns, stds=5):
    for col in [col for col in dataset.columns if col not in ignore_columns]:
        series  = dataset[col]
        dataset = dataset[~(np.abs(series-series.mean()) > stds*series.std())]
        return datasetc


def Principal_components(data1, data2, components):
    pca_model = PCA( n_components = components)
    X_train_reduc = pca_model.fit_transform(data1)
    X_test_pca    = pca_model.fit_transform(data2)
    X_train_reduc = pd.DataFrame( X_train_reduc, columns = ['PC_{}'.format(i) for i in range(components)])    
    X_test_pca = pd.DataFrame( X_test_pca, columns = ['PC_{}'.format(i) for i in range(components)])    

    return X_train_reduc, X_test_pca


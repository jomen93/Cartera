#  ===========================================================================
#  @file:   outliers.py
#  @brief:  Module to remove the outliers
#  @author: Johan Mendez
#  @date:   14/05/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================

from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

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
		return dataset

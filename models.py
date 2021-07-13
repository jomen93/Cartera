#  ===========================================================================
#  @file:   pipeline.py
#  @brief:  Main program
#  @author: Johan Mendez
#  @date:   08/07/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================
import numpy as np 

import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn import svm

def XgboostClassifier(X_train, y_train, X_test, y_test, X_val):
	
	params = {"booster":"dart", 
          "eta": 0.05, 
          "alpha":0.5,
          "gamma":5,
          # "silent":1,
          "objective":"binary:logistic",
          "eval_metric":"aucroc",
          "colsample_bytree":0.8,
          "subsample":1,
          "use_label_encoder":False,
          "nthread":4
        }


	xgb_model = XGBClassifier(**params)
	print("XgboostClassifier model")
	xgb_model.fit(X_train, 
              y_train,
              eval_set=[(X_train, y_train),(X_test, y_test)],
              eval_metric=["error", "auc"],
              early_stopping_rounds=5
             )
	
	results = xgb_model.evals_result()

	return xgb_model.predict(X_val), results

def unbalanced_randomforest():
	print("Process Started ...")
	print(" ")
	



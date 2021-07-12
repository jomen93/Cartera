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
	xgb_model.fit(X_train, 
              y_train,
              eval_set=[(X_train, y_train),(X_test, y_test)],
              eval_metric=["error", "auc"],
              early_stopping_rounds=5
             )

	return xgb_model.predict(X_val)

def Logistic_Regression(X_train, y_train, X_val):
	clf = LogisticRegression(random_state=11).fit(X_train, y_train)
	return clf.predict(X_val)

class TSVMClassifier(BaseEstimator, ClassifierMixin):
	"""docstring for TwinSVMClassifier"""
	def __init__(self, Epsilon1=0.1, Epsilon2=0.1):
		super(TSVMClassifier, self).__init__()
		self.Epsilon1 = Epsilon1
		self.Epsilon2 = Epsilon2

	def Twin_plane_1(R,S,C1,Epsi1, regulz1):
		StS = np.dot(S.T,S)

	def fit(self, X, y):
		data = sorted(zip(y, X), key=lambda pair: pair[0], reverse=True)
		total_data = np.array([np.array(x) for y,x in data])
		A = np.array([np.array(x) for y,x in data if (y==1)])
		B = np.array([np.array(x) for y,x in data if (y==0)])
		
		xcenpos = np.true_divide(sum(A), len(A))
		xcenneg = np.true_divide(sum(A), len(A))

		rcenpos = 0
		rcenneg = 0

		for a in A:
			if rcenpos < np.linalg.norm(a-xcenpos):
				rcenpos = np.linalg.norm(a-xcenpos)
		for b in B:
			if rcenneg < np.linalg.norm(a-xcenneg):
				rcenneg = np.linalg.norm(a-xcenneg)

		self.xcenpos = xcenpos
		self.xcenneg = xcenneg
		self.rcenpos = rcenpos
		self.rcenneg = rcenneg

		m1 = A.shape[0]
		m2 = B.shape[0]

		e1 = -np.ones((m1,1))
		e2 = -np.ones((m2,1))

		S = np.hstack((A, -e1))
		R = np.hstack((B, -e2))


		



		


import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier

from sklearn.grid_search import GridSearchCV

def XgboostClassifier(X_train, y_train, X_test, y_test, X_val):
	
	params = {"booster":"dart", 
          "eta": 0.05, 
          "alpha":0.5,
          "gamma":5,
          # "silent":1,
          "objective":"binary:logistic",
          "eval_metric":"auc",
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


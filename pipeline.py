#  ===========================================================================
#  @file:   pipeline.py
#  @brief:  Main program
#  @author: Johan Mendez
#  @date:   13/05/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================


from connect_database import connect
import pandas as pd
from sklearn.metrics import (mean_squared_log_error, 
                             mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             accuracy_score,
                             recall_score,
                             f1_score, precision_score,
                             plot_confusion_matrix,
                             roc_curve, auc,confusion_matrix,
                             cohen_kappa_score
                            )
import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier

import Data_cleaning
import split_data

download_data = False

# Database server credentials
server   = "carterasvr.database.windows.net"
database = "cartera"
username = "consulta"
password = "D4t4b1z2.123"


if download_data == True:
     # Database conections
     cnxn = connect(server, database, username, password)

     # SQL request
     sql = ("""SELECT HIST_CARTERA.*, REGIONES.ID_REGION_NATURAL, REGIONES.NOMBRE AS REGION
     FcROM HIST_CARTERA LEFT JOIN
          (SELECT DIM_GEOGRAFIA.ID_GEOGRAFIA, DIM_GEOGRAFIA.ID_REGION_NATURAL, DRN.NOMBRE
     FROM DIM_GEOGRAFIA
     LEFT JOIN DIM_REGION_NATURAL DRN on DIM_GEOGRAFIA.ID_REGION_NATURAL = DRN.ID_REGION_NATURAL) REGIONES
     ON HIST_CARTERA.ID_GEOGRAFIA = REGIONES.ID_GEOGRAFIA""")

     # Download the data
     data = pd.read_sql(sql, cnxn)

     data.to_pickle("data.ftr")
else:
     print("Loading data ...")
     data = pd.read_pickle("data.ftr")

print("Cleaning data ...")
X, y = Data_cleaning.Cleaning(data)
print("Tamaño de la matriz de entrenamiento = {}".format(X.shape))
print("Tamaño del vector objetivo= {}".format(y.shape))

# Split data
print("Spliting data ...")
X_train, X_test, X_val, y_train, y_test, y_val = split_data.split_stratified(X, y)


print("% de datos en el conjunto de entrenamiento = {:.2f}".format(len(X_train)/len(X)*100))
print("% de datos en el conjunto de validación    = {:.2f}".format(len(X_val)/len(X)*100))
print("% de datos en el conjunto de test          = {:.2f}".format(len(X_test)/len(X)*100))


# Model Construction
params = {"booster":"gbtree", 
          "eta": 0.05, 
          "alpha":0.5,
          "gamma":5,
          "silent":1,
          "objective":"binary:logistic",
          "eval_metric":"auc",
          "colsample_bytree":0.8,
          "subsample":1,
          "use_label_encoder":False,
         }

xgb_model = XGBClassifier()
xgb_model.fit(X_train, 
              y_train,
              eval_set=[(X_train, y_train),(X_test, y_test)],
              eval_metric=["error", "auc"],
              early_stopping_rounds=5
             )

print("Accuracy:  {}".format(accuracy_score(y_test, xgb_model.predict(X_test))))
print('Recall:    {}'.format(recall_score(y_test, xgb_model.predict(X_test))))
print('F1-Score:  {}'.format(f1_score(y_test, xgb_model.predict(X_test))))
print('Precision: {}'.format(precision_score(y_test, xgb_model.predict(X_test), zero_division="warn")))
print('Kappa:     {}'.format(cohen_kappa_score(y_test, xgb_model.predict(X_test))))




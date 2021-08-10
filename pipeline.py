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
import pickle5 as pickle
import warnings

import Data_cleaning
import split_data
import metrics
import models

warnings.filterwarnings("ignore")

# Flag to download the data
download_data = False



if download_data == True:
     
     # Database server credentials
     server   = "carterasvr.database.windows.net"
     database = "cartera"
     username = "consulta"
     password = "D4t4b1z2.123"

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
     name_pickle = "data.ftr"
     
     with open("data.ftr", "rb") as fh:
          data = pickle.load(fh)
     
     data = pd.DataFrame(data)



print("Cleaning data ...")
X, y = Data_cleaning.Cleaning(data)
print("Shape Training data = {}".format(X.shape))
print("Shape Objective vector = {}".format(y.shape))

# Split data
print("Spliting data ...")
X_train, X_test, X_val, y_train, y_test, y_val = split_data.split_stratified(X, y)


print("% train data      = {:.2f}".format(len(X_train)/len(X)*100))
print("% validation data = {:.2f}".format(len(X_val)/len(X)*100))
print("% test data       = {:.2f}".format(len(X_test)/len(X)*100))


# Model Construction
print("Training model ...")
y_pred, results = models.XgboostClassifier(X_train, y_train, X_test, y_test, X_val)

print("Calculating metrics ...")

metrics.metrics(y_val, y_pred)
metrics.plot_confusion_matrix(y_val, y_pred)
metrics.plot_XGboost(results)



#  ===========================================================================
#  @file:   pipeline.py
#  @brief:  Main program
#  @author: Johan Mendez
#  @date:   13/08/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================

import pandas as pd
import numpy as np 
import warnings
import pickle5 as pickle
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt 
warnings.filterwarnings("ignore")

import Data_cleaning
import parameter_reduction
import models 

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

# Splititng data
print("Spliting data ...")
Id_train, X_train, Y_train, Id_test, X_test, Y_test = Data_cleaning.Cleaning_time(data)


# put labels in day vector
Y_train_fin = np.zeros( (Y_train.shape[0], 91) )
Y_trainvals = Y_train['PAYDAY'].tolist()

for each_point, each_day in enumerate(Y_trainvals):
    Y_train_fin[ each_point, each_day ] = 1
    
Y_test_fin = np.zeros( (Y_test.shape[0], 91) )
Y_testvals = Y_test['PAYDAY'].tolist()

for each_point, each_day in enumerate(Y_testvals):
    Y_test_fin[ each_point, each_day ] = 1

main_components = 30
X_train_pca, X_test_pca = parameter_reduction.Principal_components( X_train, X_test, main_components )

predict_range = Y_train_fin.shape[1]


timer_model = models.NN(main_components, predict_range)

saving_callback = ModelCheckpoint( "model.ckpt", save_weights_only = True, verbose = 1)

hist_train = timer_model.fit( x = X_train_pca, y = Y_train_fin,
                             validation_data = [X_test_pca, Y_test_fin], batch_size = 128,
                             epochs = 100, callbacks = [saving_callback] )

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(121)
ax.plot( hist_train.history['loss'], "r-", label="loss")
ax.plot( hist_train.history['val_loss'], "b-", label="val loss")
ax.grid(True)
plt.legend()
ax2 = fig.add_subplot(122)
ax2.plot( hist_train.history['accuracy'], label="Accuracy")
ax2.plot( hist_train.history['val_accuracy'], label="val Accuracy")
ax2.grid(True)
plt.legend()
plt.show()













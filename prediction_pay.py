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
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('classic')
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

import Data_cleaning
import parameter_reduction
import models 
import metrics

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
                             epochs = 500, callbacks = [saving_callback] )

# save performance curves
print("saving performance curves ...")
metrics.performance_curve(hist_train)


# prediction over contract

all_dates = Id_test.merge( data[['ID_FECHA_CONSULTADA', 'NUMERO_CONTRATO', 'FECHA_REGISTRO_CARTERA', 'FECHA_COBRO',
              'FECHA_PAGO', 'FECHA_RECAUDO', 'FECHA_INICIO_CONTRATO']], how = 'left' )

all_dates[ 'CHARGEDAY' ] = (all_dates['FECHA_COBRO'] - all_dates['FECHA_REGISTRO_CARTERA'])/np.timedelta64(1, 'D')

some_contract = 1432
real_drop = Y_test[ some_contract : some_contract + 1]['PAYDAY'].tolist()[0]
proba_intime = timer_model.predict_proba( X_test_pca[some_contract : some_contract + 1] )[0]
pred_drop = np.argmax( proba_intime )
plt.figure(figsize=(30, 5))
plt.plot(proba_intime, color = 'navy', label = 'dia probabilidad pago = {} '.format(pred_drop))
plt.plot([real_drop, real_drop], [0, np.max(proba_intime)], color='limegreen', label='dia pago actual = {}'.format(real_drop))
plt.legend(loc = 'best')
plt.xticks( np.arange(0, 91, 1), rotation=90)
plt.xlim( [0, 90] )
plt.grid()
plt.show()

print("Making predictions ...")
for ijk, some_contract in enumerate(range(X_test_pca.shape[0])):

	proba_intime = list(timer_model.predict_proba( X_test_pca[some_contract : some_contract + 1] )[0])
	date_axis = [ all_dates[some_contract:some_contract+1]['FECHA_REGISTRO_CARTERA'].tolist()[0]
				 + np.timedelta64(some_num_day, 'D') for some_num_day in range( 91 ) ]

	dis_contract = [ all_dates[ some_contract : some_contract + 1]['ID_CONTRATO'].tolist()[0] ]*len(date_axis)

	result_a = pd.DataFrame(list( zip( dis_contract, date_axis, proba_intime ) ),
							columns = ['ID_CONTRATO', 'FECHA_PREDICCION', 'PROBABILIDAD_PAGO'] )

	if ijk == 0:
		fin_result = result_a.copy()
	else:
		fin_result = pd.concat( [fin_result, result_a], axis = 0 )

print("Saving csv ...")
fin_result.to_csv('Probabilidad_de_Pago_.csv', index = False, encoding = 'utf-8')


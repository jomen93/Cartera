#  ===========================================================================
#  @file:   Data_cleaning.py
#  @brief:  All data operation to model
#  @author: Johan Mendez
#  @date:   20/05/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================

import pandas as pd
import numpy as np 
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, LabelEncoder)
from time import time 

import parameter_reduction

def Cleaning(data):

	print("	Creating synthetic variables ...")
	# Definition "FECHA_CONSULTADA"
	data["FECHA_CONSULTADA"] = pd.to_datetime(data["ID_FECHA_CONSULTADA"].astype("int").astype("str"), format = "%Y-%m-%d")

	# Definition "DIAS_CONOCIMIENTO_FACTURA"
	data["DIAS_CONOCIMIENTO_FACTURA"] = (data["FECHA_CONSULTADA"] - data["FECHA_REGISTRO_CARTERA"])
	data["DIAS_CONOCIMIENTO_FACTURA"] = data["DIAS_CONOCIMIENTO_FACTURA"]/np.timedelta64(1, "D")

	# Definition "DIAS MORA"
	data["DIAS_MORA"] = (data["FECHA_PAGO"] - data["FECHA_COBRO"])/np.timedelta64(1, "D")
	# Definition auxiliary column
	data["fixem"] = 0
	# Filling missing values with one "Mora" day
	data.at[data["DIAS_MORA"].isna(), "fixem"] = 1
	# Calculation from the date consulted 
	data.at[data["fixem"]==1, "DIAS_MORA"] = (data[data["fixem"]==1]["FECHA_CONSULTADA"] - data[data["fixem"]==1]["FECHA_COBRO"])/np.timedelta64(1, "D")
	# Discard the provisional column
	data = data.drop(columns="fixem")

	# Definition "GESTION COBRO"
	data["GESTION_COBRO"] = (data["DIAS_MORA"] > 30).astype(int)
	# The name of response variable is changed
	# data.rename(columns={"GESTION_COBRO": "RESPONSE"}, inplace=True)

	# Definition "SALDADA"
	data["SALDADA"] = (~data["FECHA_PAGO"].isna()).astype("int")

	# Outliers selection
	print("	Outliers selection ...")
	Outliers_columns = ["CANTIDAD_CUOTAS_PAGADAS", "CANTIDAD_FACTURAS", "DIAS_MORA"]
	# data[Outliers_columns] = parameter_reduction.Remove_Outliers(data[Outliers_columns])

	keys = ["NUMERO_CONTRATO", "TIPO_IDENTIFICACION", "IDENTIFICACION", "SEXO", 
			"INGRESO", "REGION", "REPUTACION_CLIENTE", "GARANTIA_COLATERAL_CLIENTE", 
			"NIVEL_RIESGO_CLIENTE", "CAPACIDAD_CLIENTE", "CAPITAL_CLIENTE",
			"CIUDAD", "PROVINCIA_ESTADO_DEPARTAMENTO", "PLAZO_PACTADO",
			"CANTIDAD_CUOTAS_PAGADAS", "CANTIDAD_CUOTAS_PENDIENTES",
	    	"PORCENTAJE_INTERES_CORRIENTE_EA", "PORCENTAJE_INTERES_MORA_EA", 
	    	"SALDO_CAPITAL_CONTRATO", "VALOR_INICIAL", "CANTIDAD_FACTURAS",
	    	"ID_CONTRATO", "ID_CLIENTE", "DIAS_MORA", "GESTION_COBRO", 
	    	"DIAS_CONOCIMIENTO_FACTURA","SALDADA"]

	data = data[keys]	

	# Encode categorical variables
	categorical_model = [
	            "TIPO_IDENTIFICACION", 
	            "SEXO", 
	            "REGION", 
	            "CIUDAD", 
	            "PROVINCIA_ESTADO_DEPARTAMENTO",
	            "REPUTACION_CLIENTE", 
	            "GARANTIA_COLATERAL_CLIENTE", 
	            "NIVEL_RIESGO_CLIENTE", 
	            "CAPACIDAD_CLIENTE", 
	            "CAPITAL_CLIENTE"
	]

	print("	Encoding categorical variables ...")
	enc = OneHotEncoder(handle_unknown='ignore')
	enc.fit(data[categorical_model])
	X = enc.transform(data[categorical_model]).toarray()

	# Numerics columns
	columns_numeric = data.select_dtypes(include=["int", "float"]).columns
	# Converting pandas.core.indexes.base.Index data type to python list
	aux_num = list(columns_numeric)
	# Categorical columns are discarded
	aux_num = [i for i in aux_num if i not in categorical_model]
	# Numerical variables
	X_n = data[aux_num]


	# Columns with NaN are ignored
	NaN_columns = X_n.columns[X_n.isna().any()].tolist()
	# Removed from the columns
	aux_num = [i for i in aux_num if i not in NaN_columns]

	aux_num.remove("ID_CONTRATO")
	aux_num.remove("ID_CLIENTE")
	X_n = data[aux_num]
	

	Xn  = X_n.drop(["GESTION_COBRO"], axis=1)

	y = X_n["GESTION_COBRO"].to_numpy()
	X = np.concatenate((X, Xn.to_numpy()),axis=1) 
	
	return X, y


def Cleaning_time(data):
	# Data Modifications
	data['PAYDAY'] = (data['FECHA_PAGO'] - data['FECHA_REGISTRO_CARTERA'])/np.timedelta64(1,'D')
	data['FECHA_INICIO_CONTRATO'] = pd.to_datetime( data['FECHA_INICIO_CONTRATO'], format = '%Y-%m-%d' )
	data['CONTRACT_LIFE'] = (data['FECHA_COBRO'] - data['FECHA_INICIO_CONTRATO'])/np.timedelta64(1,'D')

	Ids = data[['ID_FECHA_CONSULTADA', 'NUMERO_CONTRATO', 'ID_CONTRATO', 'FECHA_REGISTRO_CARTERA', 'ID_CLIENTE']]
	data = data.drop( columns = ['ANIO_CARTERA', 'MES_CARTERA', 'ANIOMES_CARTERA',
	                'TIPO_IDENTIFICACION', 'IDENTIFICACION', 'FECHA_INICIO_CONTRATO', 'FECHA_REGISTRO_CARTERA',
	                'FECHA_COBRO', 'FECHA_PAGO', 'FECHA_RECAUDO', 'ID_CLIENTE', 'EN_MORA_CONTRATO_ACTUALMENTE', 'ID_GEOGRAFIA',
	                                            'CIUDAD', "REGION"] )

	cat_variables = ['SEXO', 'REPUTACION_CLIENTE', 'GARANTIA_COLATERAL_CLIENTE', 'NIVEL_RIESGO_CLIENTE',
	                'CAPACIDAD_CLIENTE', 'CAPITAL_CLIENTE', 'PROVINCIA_ESTADO_DEPARTAMENTO',
	                'EN_MORA_CONTRATO_ALGUNA_VEZ' ]


	print( 'Encoding categorical variables ...' )
	t1 = time()
	for some_feat in cat_variables:
		some_labenc = LabelEncoder()
		data[some_feat] = data[some_feat].fillna( 'OTRO' )
		data[some_feat] = some_labenc.fit_transform( data[some_feat] )
		data = pd.concat( [data, pd.get_dummies( data[some_feat], prefix = some_feat )], axis = 1 )
		data = data.drop( columns = some_feat ) 
	t2 = time()

	print( 'Encoding finished. Time = {:.2f} sec'.format(t2 - t1) )
    	
	data['Nivel_Retraso'] = 'Baja'
	data.at[data['PAYDAY'] > 30, 'Nivel_Retraso'] = 'Media'
	data.at[data['PAYDAY'] > 60, 'Nivel_Retraso'] = 'Alta'

	data = data.reset_index().drop( columns = 'index' ).reset_index().rename( columns = {'index' : 'Id_Row'} )

	data.at[ data['PAYDAY'] > 90, 'PAYDAY' ] = 90

	data = data[ ~data['PAYDAY'].isna() ]
	data['PAYDAY'] = data['PAYDAY'].astype('int')
	data_low = data[ data['Nivel_Retraso'] == 'Baja' ]
	data_mid = data[ data['Nivel_Retraso'] == 'Media' ]
	data_high = data[ data['Nivel_Retraso'] == 'Alta' ]
    
	data_low_train = data_low.sample( 850 )
	data_mid_train = data_mid.sample( 850 )
	data_high_train = data_high.sample( 850 )
    
	data_train = pd.concat( [data_low_train, data_mid_train, data_high_train], axis = 0 )
	data_train = data_train.sample( frac = 1 )
	data_test = data[ ~data['Id_Row'].isin( data_train['Id_Row'] ) ]
    
	Id_train = data_train[['Id_Row', 'ID_FECHA_CONSULTADA', 'NUMERO_CONTRATO', 'ID_CONTRATO']]
	Y_train = data_train[['Nivel_Retraso', 'PAYDAY']]
	X_train = data_train.drop( columns = ['Id_Row', 'ID_FECHA_CONSULTADA', 'NUMERO_CONTRATO',
                                          'ID_CONTRATO', 'Nivel_Retraso', 'PAYDAY'] )
	Id_test = data_test[['Id_Row', 'ID_FECHA_CONSULTADA', 'NUMERO_CONTRATO', 'ID_CONTRATO']]
	Y_test = data_test[['Nivel_Retraso', 'PAYDAY']]
	X_test = data_test.drop( columns = ['Id_Row', 'ID_FECHA_CONSULTADA', 'NUMERO_CONTRATO',
                                          'ID_CONTRATO', 'Nivel_Retraso', 'PAYDAY'] )
    
	Id_train = Id_train.reset_index().drop( columns = 'index' )
	Y_train = Y_train.reset_index().drop( columns = 'index' )
	X_train = X_train.reset_index().drop( columns = 'index' )
	Id_test = Id_test.reset_index().drop( columns = 'index' )
	Y_test = Y_test.reset_index().drop( columns = 'index' )
	X_test = X_test.reset_index().drop( columns = 'index' )

	return Id_train, X_train, Y_train, Id_test, X_test, Y_test



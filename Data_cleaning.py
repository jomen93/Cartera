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
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)

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

	# Outliers selection
	print("	Outliers selection ...")
	Outliers_columns = ["CANTIDAD_CUOTAS_PAGADAS", "CANTIDAD_FACTURAS", "DIAS_MORA"]
	X_n[Outliers_columns] = parameter_reduction.Remove_Outliers(X_n[Outliers_columns])

	NaN_columns = X_n.columns[X_n.isna().any()].tolist()
	aux_num = [i for i in aux_num if i not in NaN_columns]
	aux_num.remove("ID_CONTRATO")
	aux_num.remove("ID_CLIENTE")
	Xn  = X_n.drop(["GESTION_COBRO"], axis=1)



	y = X_n["GESTION_COBRO"].to_numpy()
	X = np.concatenate((X, Xn.to_numpy()),axis=1) 
	
	return X, y







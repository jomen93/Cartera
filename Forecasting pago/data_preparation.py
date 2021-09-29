
import pickle5 as pickle
import pandas as pd
import numpy as np

def read_data(filename):
	with open(filename, "rb") as fh:
		data = pickle.load(fh)
	return data


def temporal_series_product(data ,product, index = False):
	
	if index == True:
		fecha_pago = data["ID_CONTRATO"].value_counts().index[product].sort_values(ascending=True)
	else:
		fecha_pago = data["FECHA_PAGO"][data["ID_CONTRATO"]==product].sort_values(ascending=True)
	fecha_pago = pd.to_datetime(fecha_pago)
	first_pay = fecha_pago.min()
	last_pay  = fecha_pago.max() 
	time_product = pd.date_range(first_pay, last_pay)
	pagos = np.zeros(len(time_product))
	labels_time = np.arange(len(time_product))
	for i in range(len(fecha_pago)):
		index_to_replace = np.where(time_product == fecha_pago.iloc[i])
		pagos[index_to_replace] = 1

	return labels_time, pagos


def series_to_network(test_size, labels_time, pagos):
	
	labels_time_train = labels_time[:-test_size]
	pagos_train = pagos[:-test_size]
	labels_time_test = labels_time[-test_size:]
	pagos_test = pagos[-test_size:]

	features_set = list()
	labels = list()

	Nt = len(pagos_train)- test_size

	for i in range(Nt, len(pagos_train)):
		features_set.append(pagos_train[i-Nt:i])
		labels.append(pagos_train[i])

	features_set, labels = np.array(features_set), np.array(labels)
	features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

	return features_set, labels

def prepare_test(test_size, labels_time, pagos):

	labels_time_train = labels_time[:-test_size]
	pagos_train = pagos[:-test_size]
	labels_time_test = labels_time[-test_size:]
	pagos_test = pagos[-test_size:]

	Nt = len(pagos_train)- test_size

	test_inputs = pagos[len(pagos) - len(pagos_test) - Nt:]
	test_inputs = test_inputs.reshape(-1,1)

	test_features = list()
	for i in range(Nt, Nt+test_size):
		test_features.append(test_inputs[i-Nt:i])

	test_features = np.array(test_features)
	test_features = np.reshape(test_features, 
		(test_features.shape[0], test_features.shape[1], 1))

	return test_features








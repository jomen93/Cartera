from datetime import timedelta
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# latex == False
# #set latex configuration
# if latex == True:
# 	mpl.rc("text", usetex=True)
# 	mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

def product_prediction_curve(data, test_size, product, predictions, index=False):
	if index == True:
		fecha_pago = data["ID_CONTRATO"].value_counts().index[product].sort_values(ascending=True)
	else:
		fecha_pago = data["FECHA_PAGO"][data["ID_CONTRATO"]==product].sort_values(ascending=True)
	fecha_pago = pd.to_datetime(fecha_pago)
	first_pay = fecha_pago.min()
	last_pay  = fecha_pago.max() 
	time_product = pd.date_range(first_pay, last_pay)

	fechas = time_product[-test_size:] + timedelta(days=test_size)

	fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
	ind_max = argrelextrema(predictions, np.greater)[0]


	prefix = "Día más probable de pago {} = "
	prefix_date = [ prefix.format(i+1) for i in range(len(ind_max)) ]

	plt.plot(fechas, predictions, "b-")
	plt.plot(fechas, predictions, "b.")
	for i in range(len(ind_max)):
	    plt.plot(fechas[ind_max[i]], predictions[ind_max[i]], "ro",label=prefix_date[i]+ "{}".format(fechas[ind_max[i]].date()),
	        color="k", markerfacecolor="g")
	
	last = data[data["ID_CONTRATO"]==product]["FECHA_PAGO"].max().date()
	plt.title(f"Predicción de pago contrato {str(int(product))} (última fecha de pago registrada = {last})")
	plt.xlabel("Tiempo[Día]")
	plt.ylabel("Probabilidad[En revisión]")
	plt.grid(True)
	plt.legend()
	plt.savefig(f"Prediction_curve {int(product)}.png", transparent=True)
	print("Última fecha de fago registrada = {}".format(data[data["ID_CONTRATO"]==product]["FECHA_PAGO"].max().date()))

# Import necessary  modules 
import data_preparation
import model
import description 
from callbacks import EarlyStoppingMinLoss
import numpy as np 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
tf.random.set_seed(11)
np.random.seed(11)
# Number of physical cores.


# Define binary data from server
filename = "data.ftr"
# Month size to predict
# test_size = 30 ----> 1 month
# test_size = 60 ----> 2 month
# test_size = 90 ----> 3 month
test_size = 90

# Read datah from pickle
print("Reading data ...")
data = data_preparation.read_data(filename)

# Extraction of payments and construction of time vectors 
# from a given product in this case we choose 11575 to debug 
# product = 11575

for cliente in range(18, 21):
	product = data["ID_CONTRATO"].value_counts().index[cliente]
	# product = 6840
	print(f"Prediction of product = {product}")
	labels_time, pagos = data_preparation.temporal_series_product(data, product)

	# Preparation to neural network
	print("Preparation of fata ...")
	print(f"Horizont days = {test_size}")
	features, labels = data_preparation.series_to_network(test_size, labels_time, pagos)

	# Set up the model
	model1 = model.LSTM1(features)
	#model1 = model.LSTM_experimental(features)

	# Adjust the model 
	print("Fitting the model ...")
	model1.fit(features, 
			   labels, 
			   epochs=10, 
			   batch_size=32,
			   callbacks=[EarlyStoppingMinLoss(patience=5)]
			   )

	# make predictions
	test_features = data_preparation.prepare_test(test_size, labels_time, pagos)
	predictions = model1.predict_proba(test_features)

	# make plot prediction
	description.product_prediction_curve(data, test_size, product, predictions)


	#print(model1.get_weights())

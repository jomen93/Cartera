import numpy as np 
import torch
from torch import nn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
from datetime import datetime
import time
import tensorflow as tf
# definicion de el modo de computo
if torch.cuda.is_available() == True:
	device = "cuda"
else:
	device = "cpu"

print(f"{device} disponible!" )


def LSTM1(features_set):

	model = Sequential()
	model.add(LSTM(units=1, return_sequences=True, input_shape=(features_set.shape[1], 1)))
	model.add(Dropout(0.2))
	model.add(LSTM(units=2, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"))
	model.add(Dropout(0.2))
	model.add(LSTM(units=4, return_sequences=True, activation="sigmoid", recurrent_activation="sigmoid"))
	model.add(Dropout(0.2))
	model.add(LSTM(units=16, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"))
	model.add(Dropout(0.2))
	model.add(LSTM(units=32, return_sequences=False, activation="sigmoid", recurrent_activation="sigmoid"))
	model.add(Dropout(0.2))
	model.add(Dense(units = 1, activation="sigmoid"))

	model.compile(
		optimizer="adam", 
		# loss="binary_crossentropy", 
		#loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True)], 
		loss=[tf.keras.losses.Poisson()], 
		metrics=["accuracy"])


	return model

def LSTM_experimental(features_set):

	model = Sequential()
	model.add(LSTM(
		units=4,
		activation="sigmoid",
		recurrent_activation="sigmoid",
		#use_bias=True, 
		#recurrent_initializer='orthogonal',
		return_sequences=True,
		input_shape=(features_set.shape[1], 1)))
	model.add(Dropout(0.1))
	model.add(LSTM(
		units=1,
		activation="sigmoid",
		recurrent_activation="softmax",
		return_sequences=False,
		input_shape=(features_set.shape[1], 1)))
	model.add(Dense(units = 1, activation="sigmoid"))

	model.compile(
		optimizer="adam", 
		loss="binary_crossentropy", 
		metrics=['accuracy'])


	return model


# Se define el modulo de RNN de forma personalizada. El modulo RNN
# tendr?? una o m??s capas RNN conectadas por una capa completamente
# conectada para convertir la salida RNN en la forma de salida 
# deseadas. Se necesita tambi??n definir una fucni??n de propagaci??n 
# hacia adelante como un m??todo de la clase, que se llamar?? forward
# Este m??todo se hace secuencialmente, pasando las entradas y el estado
# oculto, inicializandose en cero. Sin embargo, Pytorch crea y calcula
# autom??ticamente la funci??n de "backpropagation"
class LSTMModel(nn.Module):
	"""
	el m??todo __init__ se inicia la instanci del modulo de Pytorch

	args:
		input_dim(int)     : numero de nodos en la capa de entrada
		hidden_dim(int)    : numero de nodos en cada capa
		layer_dim(int)     : numero de nodos en la red
		output_dim(int)     : numero de nodos de salida de la red
		dropout_prob(float): Probabilidad de descarte de los nodos
	"""
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
		super(LSTMModel, self).__init__()
		# Definicion del numero de capas y los nodos de capa
		self.hidden_dim = hidden_dim 
		self.layer_dim = layer_dim 

		# Capas RNN
		self.lstm = nn.LSTM(
			input_dim,
			hidden_dim,
			layer_dim,
			batch_first=True,
			dropout=dropout_prob
			)
		# Capa completamente conectada
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		"""
		Toma el tensor de entrada x y hace la propagaci??n hacia adelante
		Args:
			x(torch.Tensor): Tensor de entrada de la forma (batch_size, sequence_length, input_dim)

		returns:
			torch.Tensor: tensor de salida de la forma (batch_size, output_dim)
		""" 

		# inicializaci??n del estando escondido para la primera entrada con ceros
		ho = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

		# inicializacion del estado de celda para la primera entrada con ceros
		co = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

		# Se quiere nada m??s hacer la backpropagation por lotes, si no se hace esto
		# se se har?? hasta el principio, uncluso por lugares ya revisados
		out, (hn, cn) = self.lstm(x, (ho.detach(), co.detach()))

		# Conversion al estado final del la forma estipulada anteriormente
		out = self.fc(out)

		return out


# Primero , se necesita tener una clase de modelo, una fucni??nm de p??rdida 
# para calcular las p??rdidas y un optimizador para actualizar los pesos en la
# red	
class Opt:
	"""
	Clase auxiliar que permite realizar el entrenamiento, la validaci??n y la 
	predicci??n

	Atributos:
		modelo(LSTMModel): Modelo creado desde la clase para la arquitectura de RNN
		loss_fn(torch.nn.modules.Loss): Funci??n de p??rdida para calcular las p??rdidas
		optimizer(torch.optim.Optimizer): funci??n de optimizaci??n para la funci??n de perdida
		val_losses(list(float)): Lista donde se almacenas los valores de la validaci??n
		last_epoch(int): N??mero de ??pocas que el modelo va a ser entrenado
	"""
	def __init__(self, model, loss_fn, optimizer):
		
		self.model        = model
		self.loss_fn      = loss_fn
		self.optimizer    = optimizer
		self.train_losses = []
		self.val_losses   = []

	def train_step(self, x, y):
		"""
		Este m??todo hace un paso de entrenamiento. Dado un tensor (x) y la variable
		objetivo (y) se activa la backpropagation, despu??s genera los valores 
		predichos (yhat) haciendo forward, calcula las p??rdidas mediante el uso 
		de la funci??n de p??rdida. Luego, calcula los gradientes haciendo backpropagation
		y actualiza los pasos llamando la funci??n step

		Args:
			x(torch.Tensor): Tensor de caracteristicas para entrenar un paso
			y(torch.Tensor): Tensor de objetivos para calcular las p??rdidas

		"""

		# definir el modelo en entrenamiento
		self.model.train()

		# Construyendo la predicci??n
		yhat = self.model(x)

		# Calculando la funci??n de p??rdida 
		loss = self.loss_fn(y, yhat)

		# Calculando los gradientes
		loss.backward()

		# Actualizaci??n de los pesos y redefinicion de los gradientes a cero
		self.optimizer.step()
		self.optimizer.zero_grad()

		# retorno del metodo
		return loss.item()

	def train(self, train_loader, val_loader, batch_size, n_epochs, n_features):
		"""
		Ejecuci??n del entrenamiento principal:
		se toma el cargador de datos (DataLoaders) para los conjuntos de 
		entrenamiento y validaci??n. Luego se lleva a cabo el entrenamiento llamado 
		de forma iterativa al m??todod train_step n_epochs veces. 
		Finalmente guarda el modelo en una ruta de archivo seleccionada
		"""

		# se puede definir cualquier ruta, por ahora se guarda al mismo nivel del este
		# archivo
		model_path = f'{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
		start_time = time.perf_counter()
		#ciclo iterativo
		for epoch in range(1, n_epochs + 1):
			batch_losses = []
			for x_batch, y_batch in train_loader:
				x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
				y_batch = y_batch.to(device)
				loss    = self.train_step(x_batch, y_batch)
				batch_losses.append(loss)
			train_loss = np.mean(batch_losses)
			self.train_losses.append(train_loss)

			with torch.no_grad():
				batch_val_losses = []
				for x_val, y_val in val_loader:
					x_val = x_val.view([batch_size, -1, n_features]).to(device)
					y_val = y_val.to(device)
					self.model.eval()
					yhat = self.model(x_val)
					val_loss = self.loss_fn(y_val, yhat).item()
					batch_val_losses.append(val_loss)
				validation_loss = np.mean(batch_val_losses)
				self.val_losses.append(validation_loss)

			#if (epoch<=10) | (epoch%50==0):
				end_time = time.perf_counter()
				elapsed = end_time- start_time
				print(f"[{epoch}/{n_epochs}] Training loss: {train_loss:.4f}\t Validation loss:{validation_loss:.4f} time ={elapsed:.4f}")

		torch.save(self.model.state_dict(), model_path) 

	def evaluate(self, test_loader, batch_size=1, n_features=1):
		"""
		M??todo que hace la evaluaci??n del modelo: Este m??todo asume que la predicci??n 
		del paso anterior est?? disponible en el momento de la predicci??n de un paso hacia 
		el futuro

		Args:
			test_loader(torch.utils.data.DataLoader): Almacena el conjunto de validaci??n
			batch_size(int): Tama??o del mini lote de entrenamiento
			n_features(int): N??mero de carater??sticas

		Returns:
			list[float]: Valores predichos por el modelo
			list[float]: valores actuales en el conjunto de validaci??n
		"""

		with torch.no_grad():
			predictions = []
			values = []
			for x_test, y_test in test_loader:
				x_test = x_test.view([batch_size, -1, n_features]).to(device)
				y_test = y_test.to(device)
				self.model.eval()
				yhat = self.model(x_test)
				predictions.append(yhat.to(device).detach().numpy())
				values.append(y_test.to(device).detach().numpy())
		return predictions, values

	def plot_loss(self):
		"""
		hace la grafica de los valores de perdida calculados para los dos cojuntos
		"""

		plt.plot(self.train_losses, label="Training loss")
		plt.plot(self.val_losses, label="Validation loss")
		plt.legend()
		plt.title("Grafica de redimiento")
		plt.xlabel("Perdida")
		plt.ylabel("Epocas")
		plt.grid(True)
		plt.savefig("Redimiento modelo nuevo")
		#plt.show()
		plt.close()





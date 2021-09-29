import datetime 
import tensorflow as tf
import numpy as np 

class EarlyStoppingMinLoss(tf.keras.callbacks.Callback):
	"""
	Stop training when the loss function is minimum or doesn´t decreasing more

	Arguments:
	patience: epochs to wait after min has been reached
	"""
	def __init__(self, patience=0):
		super(EarlyStoppingMinLoss, self).__init__()
		self.patience = patience
		# variables to save the best weights in the minimum loss
		self.best_weights = None

	def on_train_begin(self, logs=None):
		# Number of epochs you have waiting when the loss isn´t loger minimal
		self.wait = 0
		# Stop epoch
		self.stopped_epoch = 0
		# Initialization of best like infinite
		self.best = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		current = logs.get("loss")
		if np.less(current, self.best):
			self.best = current
			self.wait = 0
			# save the best weights if the current result is better
			self.best_weights = self.model.get_weights()
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Get the best weights from the end of the best epoch")
				self.model.set_weights(self.best_weights)

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
			print(f"Epoch {self.stopped_epoch+1} Early stopped")








import numpy as np


def cost_function(y_pred, y_true):
	"""
	Calculate the cost (sum of squared erros) from a data point. Data is a 1D array of expected output (matches shape of model outputs)
	"""
	return np.sum(np.square(y_true - y_pred))

class Network:
	def __init__(self):
		"""
		Initialize network as a list of layer objects
		"""
		self.net = list()
		self.n_layers = 0

	def get_output(self):
		"""
		Get output layer of the network
		"""
		return self.net[self.n_layers-1].outputs

	def add_layer(self, input_size, n):
		"""
		Add a layer of n units to the network that can take input_size inputs
		"""
		layer = Layer(n_inputs=input_size, n_units=n)
		self.net.append(layer)
		self.n_layers += 1

	def construct_network(self, n_inputs, config):
		"""
		Construct a network from a 1D array config that can take in n_inputs.
			config: configuration of network (1D array that tells the number of neurons in each layer)
		"""
		# loop through the array and add layer 1 by 1
		for i in range(len(config)):
			# first layer takes in n_inputs inputs
			if i == 0:
				self.add_layer(n_inputs, config[0])
			# other layers takes in number of inputs equal to number of units in the previous layer
			else:
				self.add_layer(config[i-1], config[i])

	def forward_propagate(self, inputs):
		"""
		Forward propagate with inputs vector. Input vector size has to match input size
			of first layer
		"""
		for i in range(self.n_layers):
			# Layer index i
			if i == 0:
				layer_input = inputs # first layer receive input
			else:
				layer_input = self.net[i-1].outputs # next layer receives output of previous one
			self.net[i].calculate_output(layer_input)
			# Use sigmoid activation for all layers, softmax for last layer
			if i != self.n_layers-1:
				self.net[i].sigmoid()
			else:
				self.net[i].softmax()

	def error_signal(self, y_true):
		"""
		Calculate the error signal of all layers (after a forward pass). Use sum of squared errors.
		Return list of errors in each layer. Each layer's error is a 1D np array.
		"""
		# initialize empty array to contain error signal vectors
		zero = np.zeros(1)
		error = [zero] * self.n_layers
		# Calculate error signal of top layer
		top_layer = self.net[-1]
		error_top = (1 - top_layer.outputs) * top_layer.outputs * (y_true - top_layer.outputs)
		error[self.n_layers-1] = error_top
		# Calculate error signal of all non-top layer
		for i in reversed(range(self.n_layers-1)):
			layer_error = np.zeros(self.net[i].n_units)
			# loop through each neuron in the layer
			for j in range(self.net[i].n_units): # neuron index j, layer i
				error_correction = np.dot(self.net[i].weights[:-1,j], error[i+1])
				layer_error[j] = (1 - self.net[i].outputs[j]) * self.net[i].outputs[j] * error_correction
			error[i] = layer_error
		return error

	def update_weights(self, error, data, learning_rate=0.1):
		"""
		Update weights according to the error signals. Use learning rate 
			to control how fast the network learns
		"""
		# update each layer
		for i in reversed(range(self.n_layers)):
			# input is input data for first layer, previous output for other layers
			if i == 0:
				input = data
			else:
				input = self.net[i-1].outputs
			# add 1 to the end of input to update bias
			input = np.append(input, [1])
			# update each neuron index j in layer i
			for j in range(self.net[i].n_units):
				self.net[i].weights[j] += input * error[i][j] * learning_rate
			
	def train(self, data, n_epoch, n_class, learning_rate=0.1, batch_size=1):
		"""
		Train the network using the data. 
			data: 2D array of data points, with columns as features and last column as the class
			n_epoch: number of epochs to train
			n_class: number of classes for the classification task
			learning_rate: rate of learning for weight updates
			batch_size: batch size for the training process
		"""
		for epoch in range(n_epoch):
			loss = 0
			# temporary storage for each batch
			batch_error = 0
			batch_data = list()
			# NEED TO AGGREGATE THE ERRORS BEFORE UPDATE
			for idx, row in enumerate(data):
				self.forward_propagate(row[:-1])
				Y_pred = self.get_output()
				Y_train = [0 for i in range(n_class)]
				Y_train[row[-1]] = 1
				loss += cost_function(Y_pred, Y_train) # calculate value of loss function
				batch_error += self.error_signal(Y_train) # accumulate the error for this batch
				batch_data.append(row) # save the data of the batch
				# update weight after each batch
				if (idx+1) % batch_size == 0 or idx == len(data)-1:
					batch_error /= batch_size # average of errors
					# update weights
					for r in batch_data:
						self.update_weights(batch_error, r[:-1], learning_rate)
					# clear storage for nexxt batch
					batch_error = 0
					batch_data.clear()
			print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, loss))



class Layer:
	def __init__(self, n_inputs, n_units):
		"""
		Initialize a layer of n_units neurons. weights is a 2D array with n_units rows and 
			n_inputs columns. biases is a 1D array of n_units
		outputs is the vector containing outputs of all n_units neurons
		"""
		self.weights = np.random.rand(n_units, n_inputs+1)
		#self.biases = np.random.rand(n_units)
		self.outputs = np.zeros(n_units)
		self.n_units = n_units # number of units

	def calculate_output(self, inputs):
		"""
		Calculate neuron activation using dot product of weights and inputs
		inputs is a vector with 1 less element then weights + an element 1 at the end. Last element of weight vector is the bias.
		"""
		if (len(inputs) == len(self.weights[0]) - 1):
			inputs = np.append(inputs, [1])
		# loop through each neuron
		for i in range(self.n_units):
			# calculate dot product of weights and inputs, plus the bias
			self.outputs[i] = np.dot(self.weights[i], inputs)
			#self.outputs[i] = np.dot(self.weights[i,:-1], inputs) + self.weights[i,-1]
	
	def sigmoid(self):
		"""
		Activate neurons using the sigmoid function
		"""
		self.outputs = 1/(1 + np.exp(-self.outputs))

	def softmax(self):
		"""
		Activate neurons using the softmax function
		"""
		self.outputs = np.exp(self.outputs)
		sum = np.sum(self.outputs)
		self.outputs = self.outputs/sum





import numpy as np

from random import seed
from random import random
 
class Network:
	def __init__(self):
		"""
		Initialize network as a list of layer objects
		"""
		self.net = list()
		self.n_layers = 0

	def add_layer(self, input_size, n):
		"""
		Add a layer of n units to the network that can take input_size inputs
		"""
		layer = Layer(n_inputs=input_size, n_units=n)
		self.net.append(layer)
		self.n_layers += 1

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
			


class Layer:
	def __init__(self, n_inputs, n_units):
		"""
		Initialize a layer of n_units neurons. weights is a 2D array with n_units rows and 
			n_inputs columns. biases is a 1D array of n_units
		outputs is the vector containing outputs of all n_units neurons
		"""
		self.weights = np.random.rand(n_units, n_inputs)
		self.biases = np.random.rand(n_units)
		self.outputs = np.zeros(n_units)
		self.n_units = n_units # number of units

	def calculate_output(self, inputs):
		"""
		Calculate neuron activation using dot product of weights and inputs
		inputs is a vector with 1 less element then weights. Last element of weight vector is the bias.
		"""
		# loop through each neuron
		for i in range(self.n_units):
			# calculate dot product of weights and inputs, plus the bias
			self.outputs[i] = np.dot(self.weights[i], inputs) + self.biases[i]
	
	def sigmoid(self):
		"""
		Activate neurons using the sigmoid function
		"""
		self.outputs = 1/(1 + np.exp(-self.outputs))
	
	def ReLu(self):
		self.output = np.maximum(0, self.outputs)

	def softmax(self):
		"""
		Activate neurons using the softmax function
		"""
		self.outputs = np.exp(self.outputs)
		sum = np.sum(self.outputs)
		self.outputs = self.outputs/sum

 
layer1 = Layer(4, 3) # 4 inputs, 3 neurons
print(layer1.weights)
print(layer1.outputs)

input = np.array([1,2,3,-4])
layer1.calculate_output(input)
print(layer1.outputs)

layer1.sigmoid()
print(layer1.outputs)



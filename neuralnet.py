import numpy as np

from random import seed
from random import random
 
class Network:

	def __init__(self):
		"""
		Initialize network as a list of layer objects
		"""
		self.net = list()

	def add_layer(self, layer):
		"""
		Add a layer object to the network
		"""
		self.net.append(layer)

class Layer:
	def __init__(self, n_inputs, n_units):
		"""
		Initialize a layer of n_units neurons. A layer is a 2D array with n_units rows and 
			n_inputs columns (weight vector of n_units neurons). Last element of weight
			vector is the bias
		output is the vector containing outputs of all n_units neurons
		"""
		self.weights = np.random.rand(n_units, n_inputs+1)
		self.outputs = np.zeros(n_units)
		self.n_units = n_units

	def calculate_output(self, inputs):
		"""
		Calculate neuron activation using dot product of weights and inputs
		inputs is a vector with 1 less element then weights. Last element of weight vector is the bias.
		"""
		# loop through each neuron
		for i in range(self.n_units):
			# calculate dot product of weights and inputs, plus the bias
			self.outputs[i] = np.dot(self.weights[i,:-1], inputs) + self.weights[i,-1]
	
	def sigmoid(self):
		"""
		Activate neurons using the sigmoid function
		"""
		self.outputs = 1/(1 + np.exp(-self.outputs))


 
layer1 = Layer(4, 3) # 4 inputs, 3 neurons
print(layer1.weights)
print(layer1.outputs)

input = np.array([1,2,3,-4])
layer1.calculate_output(input)
print(layer1.outputs)

layer1.sigmoid()
print(layer1.outputs)



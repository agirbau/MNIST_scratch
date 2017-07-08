# print "Hello world"
# TODO:
# Somewhat like this:
# 1) Read images
# 2) Layer definition
# 3) Backpropagation
# 4) Convolution operation

# 1) simple neural net example
from numpy import exp, array, random, dot

# We have a set of training inputs and their corresponding outputs.
# I will have to try this for a XOR. In this case the training set
# is the same as the testing set (the results are always the same)
# training_set_inputs = array([[0,0],[1,0],[0,1],[1,1]])
# training_set_outpts = array([[0,1,1,0]]).T
training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_set_outpts = array([[0,1,1,0]]).T

class NeuralNetwork():
	def __init__(self)
		# We want the same random seed to be able to compare results
		random.seed(1)
		
		# We model a single neuron, with 3 inputs and 1 ouput.
		# We initialize the neuron with random weights (into a 3x1 matrix)
		# Values go from -1 to 1 with a 0 mean
		self.synaptic_weights = 2*random.random((3,1))-1

	
if __name__=="__main__"
	# Initialize 1 neural network
	neural_network = NeuralNetwork()
	
	print "Random starting synaptic weights: "
	print neural_network.synaptic_weights
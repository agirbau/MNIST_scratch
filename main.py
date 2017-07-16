# print "Hello world"
# TODO:
# Somewhat like this:
# 1) Read images
# 2) Layer definition
# 3) Backpropagation
# 4) Convolution operation

# 1) simple neural net example
from numpy import exp, array, random, dot, absolute
import matplotlib.pyplot as plt

# We have a set of training inputs and their corresponding outputs.
# I will have to try this for a XOR. In this case the training set
# is the same as the testing set (the results are always the same)
# training_set_inputs = array([[0,0],[1,0],[0,1],[1,1]])
# training_set_outputs = array([[0,1,1,0]]).T
training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_set_outputs = array([[0,1,1,0]]).T

# Todo: Every layer (which are objects) has 3 different functions: 1) Forward, Derivative respect to the input, derivate respect to paramaters
# Forward, backward derivatives, gradients

class NeuralNetwork():
    def __init__(self):
        # We want the same random seed to be able to compare results
        random.seed(1)
        
        # We model a single neuron, with 3 inputs and 1 ouput.
        # We initialize the neuron with random weights (into a 3x1 matrix)
        # Values go from -1 to 1 with a 0 mean
        self.synaptic_weights = 2*random.random((3,1))-1

    # Sigmoid function. This is the activation function of the neuron.
    # We pass the sum of the synaptic weights to this function in order to check if the neuron fires or not.
    def __sigmoid(self,x):
        sigmoid = 1 / (1 + exp(-x))
        return sigmoid
        
    # Instead of computing the derivative we just define it here.
    # Check this when backpropagating.
    def __sigmoid_derivative(self,x):
        sigmoid_derivative = x * (1-x)
        #print x
        #print x * (1-x)
        return sigmoid_derivative
        
    # Define the training of the network
    def train(self,training_set_inputs,training_set_outputs,iterations):
        for i in xrange(iterations):
        
            # 1.The input goes through the network. For each neuron we have to compute its result.
            outputs = self.think(training_set_inputs)
            #print outputs
            
            # 2.Compute the error between the prediction and the real tag.
            # In this case we define the error as the difference between the output and the tag
            # (in some other applications the error consists on cross entropy or something like that)
            errors = training_set_outputs - outputs
            if i % 100 == 0:
                errors_abs = absolute(errors)
                errors_rep = sum(errors_abs)
                print errors_rep       
                plt.scatter(i/100,errors_rep)	
                plt.pause(0.05)				
            
			
            # 3.Backpropagate the error and adjust weights
            adjustment = self.backpropagation(training_set_inputs,outputs,errors)
            #adjustment = dot(training_set_inputs.T, errors * self.__sigmoid_derivative(outputs))
			
            # 4.Adjust the weights
            self.synaptic_weights += adjustment
            
        
    # Pass inputs through the net (in this case, 1 neuron)
    def think(self,inputs):
        activation = self.__sigmoid(dot(inputs, self.synaptic_weights))
        #print inputs
        #print self.synaptic_weights
        #print dot(inputs,self.synaptic_weights)
        return activation
    
    def backpropagation(self,inputs,outputs,errors):
        error_derivate = errors * self.__sigmoid_derivative(outputs)
        adjustment = dot(inputs.T, error_derivate)
        return adjustment
        #print error_derivate
	
class MultiLayerNeuralNetwork():
	def __init__(self):
		random.seed(1)
		self.l1_synaptic_weights = 2 * random.random((3,1)) - 1 # 3 inputs
		self.l2_synaptic_weights = 2 * random.random((1,1)) - 1 # 1 input from previous layer
	
	def __sigmoid(self,sum_inputs):
		sigmoid = 1 / (1+exp(-sum_inputs))
		return sigmoid
	
	def __sigmoid_derivative(self,x):
		sigmoid_derivative = x * (1-x)
		return sigmoid_derivative
	
	def think(self,input,weights):
		sum_inputs = dot(input,weights)
		activations = self.__sigmoid(sum_inputs)
		return activations

	def backpropagation(self,input,output,error):
		der_out = error * self.__sigmoid_derivative(output)
		# print der_out
		# print input.T
		adjustment = dot(input.T, der_out)
		return adjustment
	
	def train(self,training_set_inputs,training_set_outputs,n_iter):
		for i in xrange(n_iter):
			# 1.Forward propagation
			# Sum weights * inputs
			# sum_inputs_l1 = sum(dot(training_set_inputs, self.l1_synaptic_weights))			
			# activations_l1 = self.__sigmoid(sum_inputs_l1)
			# print "Activations old: "
			# print activations_l1

			activations_l1 = self.think(training_set_inputs,self.l1_synaptic_weights)			
			# print "Activations: "
			# print activations_l1

			# sum_inputs_l2 = sum(dot(activations_l1, self.l2_synaptic_weights))
			# activations_l2 = self.__sigmoid(sum_inputs_l2)
			# print activations_l2

			activations_l2 = self.think(activations_l1,self.l2_synaptic_weights)


			# 2.Error computation
			error = activations_l2 - training_set_outputs
			# print error
			if i % 100 == 0:
				errors_abs = absolute(error)
				errors_rep = sum(errors_abs)
				print errors_rep       
				plt.scatter(i/100,errors_rep)	
				plt.pause(0.05)		

			# 3.Backpropagation
			# print activations_l1
			# print activations_l2
			adj_l1 = self.backpropagation(activations_l1,activations_l2,error)
			# print adj_l1

			adj = self.backpropagation(training_set_inputs,activations_l1,adj_l1)
			# weight update
			self.l2_synaptic_weights += adj_l1
			self.l1_synaptic_weights += adj 
		
	
	
if __name__ == "__main__":
	neural_network = MultiLayerNeuralNetwork()
	
	print "Initial weights L1: "
	print neural_network.l1_synaptic_weights
	print "Initial weights L2: "
	print neural_network.l2_synaptic_weights
	
	training_set_inputs = array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T
	print training_set_inputs
	print training_set_outputs
	
	n_iter = 10000
	neural_network.train(training_set_inputs,training_set_outputs,n_iter)
	
	
	
if __name__=="__L1__": #__main__
    #print "Hello world"
    # Initialize 1 neural network
    neural_network = NeuralNetwork()
    
    # print "Random starting synaptic weights: "
    # print neural_network.synaptic_weights
    
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T
    n_iter = 10000
    
    plt.axis([0,100,0,3])
    plt.ion() #Interactive plotting
    neural_network.train(training_set_inputs,training_set_outputs,n_iter)
    plt.show()
	
    # print "New synaptical weights"
    # print neural_network.synaptic_weights

    test_input = array([[1,0,0]])
    res = neural_network.think(test_input)
    print res

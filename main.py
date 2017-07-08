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
# training_set_outputs = array([[0,1,1,0]]).T
training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_set_outputs = array([[0,1,1,0]]).T

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
            error = training_set_outputs - outputs
            
            # 3.Backpropagate the error and adjust weights
            
        
    # Pass inputs through the net (in this case, 1 neuron)
    def think(self,inputs):
        activation = self.__sigmoid(dot(inputs, self.synaptic_weights))
        #print inputs
        #print self.synaptic_weights
        #print dot(inputs,self.synaptic_weights)
        return activation
	
if __name__=="__main__":
    #print "Hello world"
    # Initialize 1 neural network
    neural_network = NeuralNetwork()
    
    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights
    
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T
    n_iter = 1
    
    neural_network.train(training_set_inputs,training_set_outputs,n_iter)
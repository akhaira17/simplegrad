from simple_nn import *
import random

class Function:
    # flattening list of lists of parameter 
    @staticmethod
    def flatten(lis):  # sourcery skip: yield-from
        """Flattens a deeply nested list"""
        for item in lis:
            if isinstance(item, list):
                for x in Function.flatten(item):
                    yield x
            else:
                yield item   
                
    @staticmethod
    def zero_grad(model):
        assert isinstance(model, MLP), 'Must be a multilayer perceptron with a parameters'
        for p in Function.flatten(model.parameters()):
            p.grad = 0.0
    
    @staticmethod        
    def compute_binary_cross_entropy(labels, output):
        epsilon = 1e-10
        return sum([Value(-y.data) * (y_hat+epsilon).log() 
                    - Value(1-y.data) * ((1-y_hat+epsilon).log()) 
                    for y, y_hat in zip(labels, output)])
            
class Neuron:
    """
    initialises a single neuron comprising weights and a bias
    calling the class object outputs the summed up value of weights 
    multiplied by input values and applying a tanh function to introduce
    nonlinearity
    """
    def __init__(self, input_size:int, activation = 'relu', dropout = 0.5):
        # list of weights that go into a neuron
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_size)]
        self.bias = Value(random.uniform(-1,1))
        self.activation = activation
        self.dropout = dropout
        self.is_training = True
        
    def set_training_mode(self, is_training):
        self.is_training = is_training
        
    def __call__(self, x:list[float]):
        # multiplies weights by input array x
        # start from bias (same as adding )
        cell = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        # apply tanh to cell (taken from simple_nn class)
        if self.activation == 'relu':
            out = cell.relu()
        elif self.activation == 'sigmoid':
            out = cell.sigmoid()
        else:
            raise ValueError(f'Invalid activation function: {self.activation}')
        
        if self.is_training:
            if random.random() < self.dropout:
                out = Value(0)
        
        return out

    def parameters(self):
        return self.weights + [self.bias]
    
    def set_parameters(self, parameters):
        # Assumes the bias is the last element in the parameters list
        self.bias = parameters[-1]
        self.weights = parameters[:-1]
    
class Layer:
    """
    initialises a layer comprising Neuron objects
    when called, will return output of final output layer
    """
    def __init__(self, input_size:int, output_size:int, activation = 'relu'):
        self.neurons = [Neuron(input_size, activation = activation) for _ in range(output_size)]
    
    def __call__(self, x:list[float]):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        
    def parameters(self):
        return [n.parameters() for n in self.neurons]
    
    def set_parameters(self, parameters):
        for neuron, neuron_parameters in zip(self.neurons, parameters):
            neuron.set_parameters(neuron_parameters)
    
    
class MLP:
    """
    initalises a multi layered perceptron comprising Layer objects of which comprise Neuron objects
    
    Args:
        nin(int): input layer
        nouts(list): hidden layers and output layer in the order they are shown
        
    Returns:
        MLP object
    """
    def __init__(self, input_size:int, layers:list[int]):
        # layout of MLP
        size = [input_size] + layers
        # initialise layers
        # create a layer between consecutive inputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(layers) - 1)]
        # apply sigmoid to the final layer
        self.layers.append(Layer(size[-2], size[-1], activation='sigmoid'))
    
    def set_training_mode(self, is_training):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.set_training_mode(is_training)
                
    def __call__(self,x:list[float]):   
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [l.parameters() for l in self.layers]
    
    def set_parameters(self, parameters):
        for layer, layer_parameters in zip(self.layers, parameters):
            layer.set_parameters(layer_parameters)
            

        
    


from simple_nn import *
import random

class Neuron:
    """
    initialises a single neuron comprising weights and a bias
    calling the class object outputs the summed up value of weights 
    multiplied by input values and applying a tanh function to introduce
    nonlinearity
    """
    def __init__(self, nin:int):
        # list of weights that go into a neuron
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))
        
    def __call__(self, x:list[float]):
        # multiplies weights by input array x
        # start from bias (same as adding )
        cell = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        # apply tanh to cell (taken from simple_nn class)
        out = cell.tanh()
        return out
        
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:
    """
    initialises a layer comprising Neuron objects
    when called, will return output of final output layer
    """
    def __init__(self, nin:int, nout:int):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x:list[float]):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    
class MLP:
    """
    initalises a multi layered perceptron comprising Layer objects of which comprise Neuron objects
    
    Args:
        nin(int): input layer
        nouts(list): hidden layers and output layer in the order they are shown
        
    self.layers creates a list of layers with a pair of consecutive inputs
    
    e.g.
    let's assume we initialise MLP with MLP(3, [4,4,1])
    it will create 3 layer objects 
    
    layer1(3 in, 4 out)
    layer2(4 in, 4 out)
    layer3(4 in, 1 out)
    
    the final layer when called will return an output layer 
    """
    def __init__(self, nin:int, nouts:list[int]):
        # layout of MLP
        size = [nin] + nouts
        # initialise layers
        # Create a layer between consecutive inputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(nouts))]
        
    def __call__(self,x:list[float]):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
        
    


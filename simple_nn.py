import math
class Value:
    """
    Value class houses methods to enable operations between objects
    and backpropagate from a node
    """
    def __init__(self, data, _children = (), _op = '', label = ''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label
    
    # core methods 
    def __add__(self, other):
        # allow scalar addition
        other = other if isinstance(other, Value) else Value(other)
        
        # define out variable to return
        out = Value(self.data + other.data, _children = (self, other), _op = '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward  
        return out
    
    def __mul__(self, other):
        # allow scalar multiplication
        other = other if isinstance(other, Value) else Value(other)
        
        out  = Value(self.data * other.data, _children = (self, other), _op = '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert(type(other) in (float, int)), "Power functions are only supporting float or ints right now"
        
        out = Value(self.data**other, _children = (self, ), _op = f'**{other}')
        
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
 
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), _op = 'e', label = 'exp')
        
        def _backward():
            self.grad = x.data * out.grad
            
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        out = self * other**-1
        return out
    
    def tanh(self):
        # https://en.wikipedia.org/wiki/Hyperbolic_functions
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, _children = (self, ), _op = 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward
        return out
            
    # reverse methods
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return self * other**-1

    # topological approach to backward propagation
    def backward(self):
        nodes = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                    
                nodes.append(node)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(nodes):
            node._backward()
                
    # output when calling object name
    def __repr__(self):
        return f'(Value: {self.data})'

    
    

        
    
    

        

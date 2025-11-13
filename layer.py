import numpy as np
class Layer:
    """ Creates a new neural network layer.
    Activations: linear, relu, sigmoid, softmax
    """
    def __init__(self, activation: str, units: int):
        self.activation = activation
        self.units = units
        return self
    
    def set_input(self, input: int):
        self._w = np.zeros(shape=(input, self.units))
        self._b = np.zeros(shape=self.units)
    
    def output(self, input):
        z = (input @ self._w) + self._b
        match self.activation:
            case "linear":
                return z
            case "sigmoid":
                return 1 / (1 + np.exp(-z))
            case "relu":
                return np.maximum(0, z)
            case "softmax":
                e_z = np.exp(z)
                return e_z / np.sum(e_z, axis=-1) 

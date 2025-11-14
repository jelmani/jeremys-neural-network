import numpy as np
class Layer:
    """ Creates a new neural network layer.
    Activations: linear, relu, sigmoid, softmax
    """
    def __init__(self, activation: str, units: int):
        self.activation = activation
        self.units = units
        self.alpha = 0.001
        return self
    
    def set_input(self, input: int):
        self._w = np.zeros(shape=(input, self.units))
        self._b = np.zeros(shape=self.units)
    
    def forward(self, input):
        self.input = input
        z = (input @ self._w) + self._b
        match self.activation:
            case "linear":
                return z
            case "sigmoid":
                self.output = 1 / (1 + np.exp(-z))
                return self.output
            case "relu":
                self.output = np.maximum(0, z)
                return self.output
            case "softmax":
                e_z = np.exp(z)
                self.output = e_z / np.sum(e_z, axis=-1)
                return self.output
            
    def backward(self, dj_da):
        match self.activation:
            case "linear":
                dj_dz = dj_da
            case "sigmoid":
                dj_dz = dj_da * (self.output * (1 - self.output))
            case "relu":
                dj_dz = dj_da * (self.output > 0)
            case "softmax":
                dj_dz = dj_da
        dj_dw = self.input.T @ dj_dz
        dj_db = np.sum(dj_dz, axis=0, keepdims=True)
        dj_da_prev = dj_dz @ self._w.T
        self._w -= self.alpha * dj_dw
        self._b -= self.alpha * dj_db
        return dj_da_prev

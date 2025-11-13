import numpy as np

class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        for i in range(len(self.layers)):
            if i > 0:
                 self.layers[i].set_input(self.layers[i-1].units)

    def compile(self, loss: str):
        match loss:
            case "MSE":
                pass
            case "BinaryCrossEntropy":
                pass
    
    def compute_mse(self, X, y_target):
        return np.square(self.predict(X) - y_target) / (2 * X.shape[0]) 
    
    def compute_binary_ce(self, X, y_target):
        pass

    def compute_sparse_se(self, X, y_target):
        pass
    
    def predict(self, X):
        a = self.layers[0].output(X)
        for i in range(1, len(self.layers)):
            a = self.layers[i].output(a)
        return a
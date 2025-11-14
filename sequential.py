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
                self.loss = self.compute_mse
            case "BinaryCrossEntropy":
                self.loss = self.compute_binary_ce
            case "SparseCrossEntropy":
                self.loss = self.compute_sparse_se
    
    def compute_mse(self, X, y_target, derivative: bool = False):
        resid = self.predict(X) - y_target
        if derivative:
            return np.mean(resid, axis=0)
        return 0.5 * np.mean(resid**2)
    
    
    def compute_binary_ce(self, X, y_target, derivative: bool = False):
        y_pred = self.predict(X)
        resid = y_pred - y_target
        if derivative:
            return np.mean(resid, axis=0)
        return np.mean(-y_target * np.log(y_pred) - (1 - y_target) * np.log(1 - y_pred))

    def compute_sparse_se(self, X, y_target, derivative: bool = False):
        y_pred = self.predict(X)
        if derivative:
            m = y_target.shape[0]
            num_classes = y_pred.shape[1]
            Y = np.zeros((m, num_classes))
            Y[np.arange(m), y_target] = 1
            return (y_pred - Y) / m
        N = y_target.astype(int).ravel()
        return np.mean(-np.log(y_pred[:, N]))
    
    def predict(self, X):
        a = self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
            a = self.layers[i].forward(a)
        return a
    
    def fit(self, X, y_target):
        self.layers[0].set_input(X.shape[1])
        # Backpropagation
        dj_da = self.loss(X, y_target, derivative=True)
        for i in range(len(self.layers) - 1, -1, -1):
            dj_da = self.layers[i].backward(dj_da)
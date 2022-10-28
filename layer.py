import numpy as np

def sigmaFunction(value):
    return 1 / (1 + np.exp(-value))

vectorize = np.vectorize(sigmaFunction)

class Layer:

    def __init__(self, nodes, prevLayer=None, bias=None, weights=None, rate=1):
        #Initialize
        self.totalNodes = nodes
        self.activation = np.zeros([self.totalNodes, 1])
        self.rate = rate
        self.costDY = None
        self.low=0.1
        self.high=1

        self.nextLayer = None
        self.prevLayer = prevLayer
        if self.prevLayer:
            self.setPrevLayer(self.prevLayer)

        self.weights = weights
        if self.weights is None:
            self.totalcostDW = 0
        else:
            self.totalcostDW = np.zeros(self.weights.shape)

        self.bias = bias
        if self.bias is None:
            self.totalcostDB = 0
        else:
            self.totalcostDB = np.zeros(self.bias.shape)
    
    def cost(self, y):
        # Cost function - Calculated the cost of the current layer
        errSq = (y - self.activation) ** 2
        return errSq.sum() / (2 * self.activation.size)

    def costDB(self):
        multiply = self.costDZ() * self.DZ_DB()
        return np.dot(multiply, np.ones(multiply.shape[1]))[np.newaxis].T

    def DY_ZY(self):
        return self.activation * (1 - self.activation)

    def DZ_DA(self):
        return self.weights.T
    
    def DZ_DB(self):
        return np.ones((self.totalNodes, 1), dtype=float)
    
    def DZ_DW(self):
        return self.prevLayer.activation.T

    def costDW(self):
        return np.dot(self.costDZ(), self.DZ_DW())

    def costDZ(self):
        return self.costDY * self.DY_ZY()

    def calcDY(self, y=None):
        if y is None:
            self.costDY = np.dot(self.nextLayer.DZ_DA(), self.nextLayer.costDZ())
        else:
            self.costDY = - (y - self.activation) / y.size
        return self.costDY

    def randomWeights(self):
        # Give random weights to first creation of layer (since none exist)
        self.weights = np.random.uniform(size=(self.totalNodes, self.prevLayer.totalNodes),
                                         low=self.low,
                                        high=self.high)

    def randomBias(self):
        # Give random bias to first creation of layer (since none exist)
        self.bias = np.random.uniform(size=(self.totalNodes, 1),
                                      low=self.low,
                                      high=self.high)
    
    def activate(self):
        # Activation function
        self.activation = vectorize(np.dot(self.weights, self.prevLayer.activation) + self.bias)
        return self.activation

    def setPrevLayer(self, layer):
        # Set previous layer
        self.prevLayer = layer
        self.prevLayer.nextLayer = self

    def addBias(self):
        #Helper to add bias to totalcostDB
        self.totalcostDB += self.costDB()
        
    def addWeights(self):
        #Helper to add weights to totalcostDW
        self.totalcostDW += self.costDW()

    def recalculateAll(self):
        # Backward propagation function to update weights and bias
        self.weights = self.weights - self.rate * self.totalcostDW
        self.bias = self.bias - self.rate * self.totalcostDB
        self.totalcostDW = np.zeros(self.weights.shape)
        self.totalcostDB = np.zeros(self.bias.shape)
   
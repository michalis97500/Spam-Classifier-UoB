import json
import numpy as np
from layer import Layer


class NeuralNetwork:

    def __init__(self, currentSettings):
        # Initialize
        self.layers = [Layer(**settings) for settings in currentSettings]
        self.link()
        self.trainFeatures = None
        self.trainLabels = None

    def calcAccuracy(self, features, labels):
        # Calculate accuracy
        prediction = self.predict(features)
        diffSQ = (labels - prediction) ** 2
        return 1 - (diffSQ.sum() / prediction.size)

        
    def predict(self, array):
        # Function required for assignemnt
        self.layers[0].activation = array.T
        for layer in self.layers[1:]:
            layer.activate()
        return self.layers[-1].activation.round(0)
    
    
    def link(self):
        #Link layer with previous layer
        for i, layer in enumerate(self.layers[1:]):
            layer.setPrevLayer(self.layers[i])


    def fwdPropagation(self, training_dataset_batch, training_labels_batch):
        #Forward propagation funcion
        accuracy = self.calcAccuracy(features=training_dataset_batch,
                                                labels=training_labels_batch)
        return accuracy

    def backPropagation(self, labels):
        # Back propagation function
        self.layers[-1].calcDY(y=labels)
        for layer in self.layers[1:-1][::-1]:
            layer.calcDY()
        for layer in self.layers[1:][::-1]:
            layer.addWeights()
            layer.addBias()
            layer.recalculateAll()

    def layerPrepare(self):
        # Check if the layers are ready for training
        # If not, assign random weights and bias
        for layer in self.layers[1:]:
            if layer.bias is None:
                layer.randomBias()
            if layer.weights is None:
                layer.randomWeights()
                
    def train(self, epochs, batchSize=100):
        # Train the network
        self.layerPrepare()
        for epoch in range(epochs):
            totalBatches = self.trainFeatures.shape[1] // batchSize + 1
            for batch in range(totalBatches):
                features, labels = self.getTrainBatch(batch, batchSize)
                accuracy = self.fwdPropagation(features, labels)
                self.backPropagation(labels)
                print(accuracy)

    def getTrainBatch(self, batch, batchSize):
        # Get a batch of training data ready for training
        begin = batch * batchSize
        end = (batch + 1) * batchSize
        return self.trainFeatures[begin:end, ], self.trainLabels[begin:end]
    
    
    def setTrainData(self, features, labels):
        #Helper function to set training data
        self.trainFeatures = features
        self.trainLabels = labels


    def save(self, filename):
        #Save the weights and bias to a json file
        datalist = list()
        for i, layer in enumerate(self.layers):
            if layer.weights is None:
                weights = [[]]
            else:
                weights = layer.weights.tolist()
            if layer.bias is None:
                bias = [[]]
            else:
                bias = layer.bias.tolist()
            data = {'layer': i,
                          'weights': weights,
                          'bias': bias}
            datalist.append(data)
        with open(filename, 'w') as F:
            json.dump(obj=datalist,
                      fp=F)

    def load(self, filename):
        #Load the weights and bias from a json file
        with open(filename, 'r') as F:
            datalist = json.load(F)
        for data in datalist:
            self.layers[data['layer']].weights = np.array(data['weights'])
            self.layers[data['layer']].bias = np.array(data['bias'])

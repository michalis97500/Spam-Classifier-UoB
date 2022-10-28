import layer
from ai import NeuralNetwork
import numpy as np



def create_classifier(learning_rate, startNode, hiddenNode, epochs1, batchSize1):
    training_full_dataset = np.genfromtxt("data/training_spam.csv", delimiter=',')
    test_full_dataset = np.genfromtxt("data/testing_spam.csv", delimiter=',')    
    training_dataset = training_full_dataset[:, 1:]
    training_labels = training_full_dataset[:, 0]
    test_dataset = test_full_dataset[:, 1:]
    test_labels = test_full_dataset[:, 0]
    layer_settings = [{'nodes': startNode, 'rate': learning_rate},
                      {'nodes': hiddenNode, 'rate': learning_rate},
                      {'nodes': 1, 'rate': learning_rate}]
    nn = NeuralNetwork(currentSettings=layer_settings)
    nn.setTrainData(features=training_dataset,labels=training_labels)
    nn.train(epochs=epochs1,batchSize=batchSize1)
    nn.predict(test_dataset)
    acc = nn.calcAccuracy(features=test_dataset,labels=test_labels).round(3)
    nn.save("data/weights_and_bias.json")
    return acc

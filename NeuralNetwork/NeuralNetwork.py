import numpy


class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # Number of nodes in each layer
        self.inNodes = inputNodes
        self.hidNodes = hiddenNodes
        self.outNodes = outputNodes

        # Learning rate
        self.learnRate = learningRate

        # Weights
        self.wih = (numpy.random.rand(self.hidNodes, self.inNodes) - 0.5)
        self.who = (numpy.random.rand(self.outNodes, self.hidNodes) - 0.5)
        pass

    def train(self):
        pass

    def query(self):
        pass


pass

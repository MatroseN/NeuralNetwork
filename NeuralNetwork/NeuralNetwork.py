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
        self.wih = numpy.random.normal(0.0, pow(self.hidNodes, -0.5), (self.hidNodes, self.inNodes))
        self.who = numpy.random.normal(0.0, pow(self.outNodes, -0.5), (self.outNodes, self.hidNodes))
        pass

    def train(self):
        pass

    def query(self):
        pass


pass

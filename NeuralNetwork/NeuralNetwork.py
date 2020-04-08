import numpy
import scipy.special as sci


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

        # Activation function is the sigmoid function
        self.activationFunction = lambda x: sci.expit(x)
        pass

    def train(self, inputsList, targetsList):
        inputs = numpy.array(inputsList, ndmin=2).T
        targets = numpy.array(targetsList, ndmin=2).T

        hiddenInputs = numpy.dot(self.wih, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        finalInputs = numpy.dot(self.who, hiddenOutputs)
        finalOutputs = numpy.dot(self.activationFunction(finalInputs))

        outputErrors = targets - finalOutputs
        hiddenErrors = numpy.dot(self.who.T, outputErrors)

        self.who += self.lr * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), numpy.transpose(hiddenOutputs))

        self.wih += self.lr * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), numpy.transpose(inputs))
        pass

    def query(self, inputsList):
        # Convert inputs to 2d array
        inputs = numpy.array(inputsList, ndmin=2).T

        # Calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        # Calculate the signals coming from the hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)

        # Calculate signals into final output layer
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        # Calculate the signals coming from the final output layer
        finalOutputs = self.activationFunction(finalInputs)

        return finalOutputs
        pass



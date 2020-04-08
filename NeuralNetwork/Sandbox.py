import NeuralNetwork

inNodes = 3
hidNodes = 3
outNodes = 3

learnRate = 0.3

neuralNet = NeuralNetwork.NeuralNetwork(inNodes, hidNodes, outNodes, learnRate)

print(neuralNet.query([1.0, 0.5, -1.5]))

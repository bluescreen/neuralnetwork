import matplotlib.pyplot
from network import NeuralNetwork
from pprint import pprint;

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

# input_nodes = 784
# hidden_nodes = 200
# output_nodes = 10
# learning_rate = 0.1

network = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

result = network.query([1.0,0.5,-1.5])
pprint(result)

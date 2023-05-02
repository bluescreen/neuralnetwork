from network import NeuralNetwork
from train import train_network, test_network

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
max_train = 10000
epochs = 3

network = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

train_network('data/mnist_train.csv', network, output_nodes,epochs, max=max_train)
performance = test_network('data/mnist_train.csv', network, max=max_train)

print("performance", '{0:.4g}'.format(performance*100))

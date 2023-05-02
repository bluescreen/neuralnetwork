
import numpy
import scipy.special

class NeuralNetwork:
    def __init__(self,input_nodes, hidden_nodes, output_nodes, learning_rate):
        print("init")
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.wih = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes, ))
        self.who = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes, ))

        self.learning_rate = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x)

    
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # backpropagation
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.learning_rate * numpy.dot((output_errors * final_outputs * 1.0 - final_outputs), numpy.transpose(hidden_outputs))
        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * 1.0 - hidden_outputs), numpy.transpose(inputs))
        pass


    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

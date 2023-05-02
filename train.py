
import numpy

def load_data(file_name):
    file = open(file_name)
    records = file.readlines()
    file.close()
    return records

def train_network(file, network,output_nodes,epochs =1,  max= 10000):
    train_data = load_data(file)

    print("train network", len(train_data))

    for e in range(epochs):
        for i, record in enumerate(train_data, start=1):
            if(i >= max): break 
            print("train epoch",e," record ", i)
            values = record.split(',')
            inputs = (numpy.asfarray(values[1:]) / 255 * 0.99) +0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(values[0])] = 0.99
            network.train(inputs, targets)

def test_network(test_file, network, max):
    scores = []
    test_data = load_data(test_file)

    for i,record in enumerate(test_data, start=1):
        if(i >= max): break 
        print("test", i)
        values = record.split(',')
        correct_label = int(values[0])
        inputs = (numpy.asfarray(values[1:]) / 255 * 0.99) +0.01
        outputs = network.query(inputs)

        label = numpy.argmax(outputs)
        if(label == correct_label):
            scores.append(1)
        else:
            scores.append(0)

    scores_array = numpy.asarray(scores)
    return scores_array.sum() / scores_array.size

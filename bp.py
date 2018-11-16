# Very useful: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# Also, https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
import matplotlib.pyplot as plt
from random import random
import math


# NOT USED: Initialize a network with arbitrary weights
def initialize_network_with_random_weights(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [round(random(), 1) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [round(random(), 1) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate the derivative of an neuron output
def sigmoid_derivative(output):
    return output * (1.0 - output)


# Print the weights
def print_weights(network):
    l1 = network[0]
    l2 = network[1]
    print_gist(
        "Weights:\nw13\tw14\tw15\tw23\tw24\tw25\tw36\tw37\tw46\tw47\tw56\tw57\n========================================")
    print_gist("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" %
          (l1[0]['h3'][0], l1[1]['h4'][0], l1[2]['h5'][0],
           l1[0]['h3'][1], l1[1]['h4'][1], l1[2]['h5'][1],
           l2[0]['o6'][0], l2[1]['o7'][0],
           l2[0]['o6'][1], l2[1]['o7'][1],
           l2[0]['o6'][2], l2[1]['o7'][2]))
    print_gist("Biases:\nb3\tb4\tb5\tb6\tb7\n==================================")
    print_gist("%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (l1[0]['h3'][-1], l1[1]['h4'][-1], l1[2]['h5'][-1],
                                            l2[0]['o6'][-1], l2[1]['o7'][-1]))


# 1: Initialize a network with the given weights
def initialize_network():
    network = list()
    hidden_layer = [{'h3': [0.1, -0.2, 0.1]}, {'h4': [0, 0.2, 0.2]}, {'h5': [0.3, -0.4, 0.5]}]
    output_layer = [{'o6': [-0.4, 0.1, 0.6, -0.1]}, {'o7': [0.2, -0.1, -0.2, 0.6]}]
    network.append(hidden_layer)
    network.append(output_layer)
    n_outputs = 2

    print("STEP 1: initialize weights and biases\n=====================")
    print_weights(network)
    print()
    return network, n_outputs


# 2: Forward pass
def forward_propagate(network, row):
    inputs = row
    print_gist("STEP 2: Calculate neurons' inputs and outputs.\n=====================")
    print_gist("Unit (j)\tInput (I_j)\tOutput (O_j)")
    for layer in network:
        new_inputs = []
        for neuron in layer:
            key = list(neuron.keys())[0]
            activation = activate(neuron[key], inputs)
            neuron['output'] = sig(activation)
            new_inputs.append(neuron['output'])
            print_gist("%s\t%.3f\t%.3f" % (key, activation, neuron['output']))
        inputs = new_inputs
    print_gist()
    return inputs


# 3. Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    print_gist("STEP 3: Calculate the error at each neuron.\n=====================")
    print_gist("Unit (j)\tError (Err_j)")

    for i in reversed(range(len(network))):
        layer = network[i]
        len_layer = len(layer)
        errors = list()
        if i != len(network) - 1: # => the hidden layer
            for j in range(len_layer):
                error = 0.0
                for neuron in network[i + 1]:
                    key = list(neuron.keys())[0]
                    error += (neuron[key][j] * neuron['delta'])
                errors.append(error)
        else: # => the outer layer
            for j in range(len_layer):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len_layer):
            neuron = layer[j]
            key = list(neuron.keys())[0]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
            print_gist("%s\t%.3f" % (key, neuron['delta']))
    print_gist()


# 4: Update network weights with error
def update_weights(network, row, l_rate):
    print_gist("STEP 4: Update weights and biases\n=====================")

    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            key = list(neuron.keys())[0]
            for j in range(len(inputs)):
                neuron[key][j] += l_rate * neuron['delta'] * inputs[j] # the weights
            neuron[key][-1] += l_rate * neuron['delta'] # the bias

    print_weights(network)
    print_gist()


# Train a network for a fixed number of epochs
def train_network(train_dataset, l_rate, n_epoch=1):
    network, n_outputs = initialize_network()
    x_epochs = []
    y_errors = []
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train_dataset:
            print_gist("STEP 1: initialize weights and biases\n=====================")
            print_gist("input row: {}. Learning rate: {}. Layers:".format(row, l_rate))
            print_weights(network)
            print_gist()
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            # if epoch == n_epoch - 1:
            #     # print("input row: {}".format(row))
            #     print()
            #     print("outputs: {}".format(outputs))
            #     print("expected: {}".format(expected))
        if epoch % 50 == 0:
            print('>epoch = %d, learning rate = %.3f, error = %.3f' % (epoch, l_rate, sum_error))
        x_epochs.append(epoch)
        y_errors.append(sum_error)
    plt.plot(x_epochs, y_errors)
    plt.title("Error vs Epoch")
    plt.xlabel("Number of epochs")
    plt.ylabel("Error")
    plt.show()


# Calculate neuron activation for an input: w.x + b
def activate(weights, inputs):
    activation = weights[-1]  # * 1; the bias value
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# The sigmoid function. That's our transfer function
def sig(activation):
    return 1.0 / (1.0 + math.exp(-activation))


def print_gist(*args):
    print(*args)
    # pass


def main():
    # NB: the last item in array is the label.
    # (nail, screw) -> (0, 1) -> (10, 01)
    # dataset = [[0.6, 0.1, 10],
    #            [0.2, 0.3, 1]]
    dataset = [[0.6, 0.1, 0]]
    l_rate = 0.1
    train_network(dataset, l_rate, n_epoch=1001)


if __name__ == "__main__":
    main()

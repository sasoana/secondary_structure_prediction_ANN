#! python3
import numpy as np
from numpy.core.tests.test_mem_overlap import xrange
from ANN import ANN
import sys
import matplotlib.pyplot as plt


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def load_dataset(filename, window_size):
    encoding = {
        "A": 0.05, "C": 0.1, "E": 0.15, "D": 0.2, "G": 0.25, "F": 0.3, "I": 0.35, "H": 0.4, "K": 0.45, "M": 0.5,
        "L": 0.55, "N": 0.6, "Q": 0.65, "P": 0.7, "S": 0.75, "R": 0.8, "T": 0.85, "W": 0.9, "V": 0.95, "Y": 1
    }

    data_file = open(filename, "r")
    proteins = []  # list of proteins, from <> to end, in the data file
    secondary_classes_overall = []  # list of classes for each protein
    data = data_file.read().splitlines(keepends=False)
    protein = []  # amino acids for one protein, one letter encoding
    secondary_classes_for_protein = []  # secondary classes for amino acids in one protein
    for idx, line in enumerate(data):
        tokens = line.split()  # split each line in amino acid and secondary structure class
        if len(tokens) == 0:
            continue
        if tokens[0] == 'end':
            # if at the end of protein, continue
            continue
        if tokens[0] == '<>':
            # skip the first line with this character
            if idx > 0:
                proteins.append(protein)
                secondary_classes_overall.append(secondary_classes_for_protein)
            if idx != len(data):
                protein = []
                secondary_classes_for_protein = []
        else:
            # if we are in the middle of reading a protein sequence, append info in the corresponding lists
            protein.append(encoding[tokens[0]])
            if tokens[1] == "h":
                actual_output = [1, 0, 0]
            if tokens[1] == "e":
                actual_output = [0, 1, 0]
            if tokens[1] == "_":
                actual_output = [0, 0, 1]
            secondary_classes_for_protein.append(actual_output)
    # add the last set of data
    proteins.append(protein)
    secondary_classes_overall.append(secondary_classes_for_protein)
    data_file.close()
    step = window_size // 2
    # add the first sequence of encoded amino acids to the final list
    X = np.array(([proteins[0][:window_size]]), dtype=float)
    # take sequences of "window_size" amino acids from each protein
    for protein in proteins:
        seqLength = len(protein)
        for i in xrange(step + 1, seqLength - step):
            X = np.concatenate((X, np.array(([protein[i - step:i + step + 1]]), dtype=float)))
    # print(X[:50])

    # add the class for the middle amino acid in the first sequence of length "window_size" -> on position step
    y = np.array(([secondary_classes_overall[0][step]]), dtype=float)
    for structure in secondary_classes_overall:
        seqLength = len(structure)
        for i in xrange(step + 1, seqLength - step):
            y = np.concatenate((y, np.array(([structure[i]]), dtype=float)))
    # print(y[:50])

    return X, y


if __name__ == "__main__":

    if (len(sys.argv)) < 5:
        print('Please run the application using the following parameters:\n' +
              'window size, batch size, epochs, learning rate, nb neurons on hidden layer 1, nb neurons on hidden layer 2... ')
        exit()
    window_size = int(sys.argv[1])
    minibatch_size = int(sys.argv[2])
    epochs = int(sys.argv[3])
    eta = float(sys.argv[4])
    layers = []
    for i in xrange(5, len(sys.argv)):
        layers.append(int(sys.argv[i]))
    X, y = load_dataset("protein-secondary-structure.train.txt", window_size)
    Xtest, ytest = load_dataset("protein-secondary-structure.test.txt", window_size)

    train_data = []
    train_labels = []
    for batch in iterate_minibatches(X, y, minibatch_size, shuffle=True):
        x_batch, y_batch = batch
        train_data.append(x_batch)
        train_labels.append(y_batch)

    valid_data = []
    valid_labels = []
    for batch in iterate_minibatches(Xtest, ytest, minibatch_size, shuffle=True):
        x_batch, y_batch = batch
        valid_data.append(x_batch)
        valid_labels.append(y_batch)
    config = [window_size]
    for layer in layers:
        config.append(layer)
    config.append(3)
    ann = ANN(eta=eta, layer_config=config, epochs=epochs, minibatch_size=minibatch_size)
    train_error, test_error = ann.evaluate(train_data, train_labels, valid_data, valid_labels, eval_train=True)

    x_axis = np.array(list(range(1, epochs + 1)))
    plt.plot(x_axis, train_error, 'r')
    plt.plot(x_axis, test_error, 'b')
    plt.axis([0, epochs, 0, 1])
    plt.show()

    # ann.forward_propagate(np.array([0.15, 0.6, 0.55, 0.45, 0.55, 0.25, 0.3], dtype=float).T)

#! python3
import numpy as np
from numpy.core.tests.test_mem_overlap import xrange
import matplotlib.pyplot as plt

from ANN import ANN


class Controller:
    def __init__(self, ann, window_size=11, batch_size=100, epochs=100, learning_rate=0.03, layer_config=[15]):
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layer_config = [window_size] + layer_config + [3]
        self.ann = ann
        self.encoding = { "spacer": 0.00,
                          "A": 0.05, "C": 0.1, "E": 0.15, "D": 0.2, "G": 0.25,
                          "F": 0.3, "I": 0.35, "H": 0.4, "K": 0.45, "M": 0.5,
                          "L": 0.55, "N": 0.6, "Q": 0.65, "P": 0.7, "S": 0.75,
                          "R": 0.8, "T": 0.85, "W": 0.9, "V": 0.95, "Y": 1
                        }
        self.train_file = "protein-secondary-structure.train.txt"
        self.test_file = "protein-secondary-structure.test.txt"

    def load_dataset(self, filename, window_size):
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
                protein.append(self.encoding[tokens[0]])
                if tokens[1] == "h":
                    actual_output = [1, 0, 0]
                if tokens[1] == "e":
                    actual_output = [0, 1, 0]
                if tokens[1] == "_":
                    actual_output = [0, 0, 1]
                secondary_classes_for_protein.append(actual_output)
        data_file.close()
        step = window_size // 2
        # add the first sequences of encoded amino acids to the final list
        X = np.array(([proteins[0][:window_size]]), dtype=float)
        first_protein = True
        # take sequences of "window_size" amino acids from each protein
        for protein in proteins:
            seq_length = len(protein)
            if seq_length < self.window_size:
                continue
            for i in xrange(0, step):
                one_input = [self.encoding['spacer']] * (step - i)
                one_input += protein[:step + i + 1]
                X = np.concatenate((X, np.array(([one_input]), dtype=float)))
            start = step
            if first_protein:
                start = step + 1
                first_protein = False
            for i in xrange(start, seq_length - step):
                X = np.concatenate((X, np.array(([protein[i - step:i + step + 1]]), dtype=float)))
            for i in xrange(0, step):
                one_input = protein[seq_length-self.window_size + i + 1:]
                one_input += [self.encoding['spacer']] * (i + 1)
                X = np.concatenate((X, np.array(([one_input]), dtype=float)))

        # add the class for the middle amino acid in the first sequence of length "window_size" -> on position step
        y = np.array(([secondary_classes_overall[0][step]]), dtype=float)
        first_protein = True
        for structure in secondary_classes_overall:
            seq_length = len(structure)
            if seq_length < self.window_size:
                continue
            for i in xrange(0, step):
                y = np.concatenate((y, np.array(([structure[i]]), dtype=float)))
            start = step
            if first_protein:
                start = step + 1
                first_protein = False
            for i in xrange(start, seq_length):
                y = np.concatenate((y, np.array(([structure[i]]), dtype=float)))

        return X, y

    def iterate_minibatches(self, inputs, targets, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - self.batch_size + 1, self.batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            yield inputs[excerpt], targets[excerpt]

    def train_and_evaluate(self):
        self.ann = ANN(eta=self.learning_rate, minibatch_size=self.batch_size, layer_config=self.layer_config, epochs=self.epochs)
        X, y = self.load_dataset(self.train_file, self.window_size)
        Xtest, ytest = self.load_dataset(self.test_file, self.window_size)
        train_data = []; train_labels = []
        for batch in self.iterate_minibatches(X, y, shuffle=True):
            x_batch, y_batch = batch
            train_data.append(x_batch)
            train_labels.append(y_batch)
        valid_data = []; valid_labels = []
        for batch in self.iterate_minibatches(Xtest, ytest, shuffle=True):
            x_batch, y_batch = batch
            valid_data.append(x_batch)
            valid_labels.append(y_batch)

        train_error, test_error = self.ann.evaluate(train_data, train_labels, valid_data, valid_labels, eval_train=True)
        self.show_errors_plot(train_error, test_error)

    def show_errors_plot(self, train_error, test_error):
        x_axis = np.array(list(range(1, self.epochs + 1)))
        plt.plot(x_axis, 100*train_error, 'r', label='training error')
        plt.plot(x_axis, 100*test_error, 'b', label='testing error')
        plt.axis([0, self.epochs, 0, 100])
        plt.xlabel('Epoch number')
        plt.ylabel('Error (%)')
        plt.title('Training and testing errors')
        plt.legend()
        plt.show()

    def split_protein(self, sequence):
        encoded_sequence = [self.encoding[letter] for letter in sequence]
        step = self.window_size // 2
        first_input = [self.encoding['spacer']] * step
        first_input += encoded_sequence[:step+1]
        X = np.array(([first_input]), dtype=float)
        for i in xrange(1, step):
            one_input = [self.encoding['spacer']] * (step - i)
            one_input += encoded_sequence[:step + i + 1]
            X = np.concatenate((X, np.array(([one_input]), dtype=float)))
        for i in xrange(step, len(sequence) - step):
            X = np.concatenate((X, np.array(([encoded_sequence[i - step:i + step + 1]]), dtype=float)))
        for i in xrange(step):
            one_input = encoded_sequence[len(encoded_sequence) - self.window_size + i + 1:]
            one_input += [self.encoding['spacer']] * (i + 1)
            X = np.concatenate((X, np.array(([one_input]), dtype=float)))

        return X

    def predict_for_protein(self, sequence):
        input_data = self.split_protein(sequence)

        output = self.ann.forward_propagate(input_data)
        result = ''
        for prediction in np.argmax(output, axis=1):
            if prediction == 0:
                result += 'h'
            if prediction == 1:
                result += 'e'
            if prediction == 2:
                result += '_'
        assert len(input_data) == len(result)

        return result

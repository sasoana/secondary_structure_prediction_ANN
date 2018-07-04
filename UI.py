#! python3
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QComboBox, QVBoxLayout, QHBoxLayout, QPushButton, \
    QPlainTextEdit, QMessageBox
from Controller import Controller


class App(QWidget):

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.window_edit = QComboBox()
        self.batch_edit = QComboBox()
        self.epochs_edit = QComboBox()
        self.eta_edit = QComboBox()
        self.layers_edit = QComboBox()
        self.layers = []
        self.nb_neurons = ['15', '25', '50', '100']
        self.train_button = QPushButton()
        self.layers_box = QHBoxLayout()

        self.init_UI()

    def init_UI(self):
        self.setFont(QFont("Arial", 18))
        info_label = QLabel('This is an application for predicting protein secondary structure ' +
                            'using an artificial neural network. Please select the desired parameters ' +
                            'and then train the network.')
        info_label.setWordWrap(True)

        # parameter labels and combo boxes
        window = QLabel('Window size')
        batch = QLabel('Batch size')
        epochs = QLabel('Number of epochs')
        eta = QLabel('Learning rate')
        layers = QLabel('Number of hidden layers')

        self.window_edit.addItems(['11', '13', '15', '25'])
        self.batch_edit.addItems(['50', '100', '200'])
        self.epochs_edit.addItems(['100', '1000', '5000', '10000'])
        self.eta_edit.addItems(['0.01', '0.02', '0.03'])
        self.layers_edit.addItems(['1', '2', '3', '4'])

        # add labels and combo boxes into grid layout
        params_box = QVBoxLayout()
        grid = QGridLayout()
        grid.setContentsMargins(10, 20, 20, 20)
        grid.addWidget(window, 1, 0)
        grid.addWidget(self.window_edit, 1, 1)
        grid.addWidget(batch, 2, 0)
        grid.addWidget(self.batch_edit, 2, 1)
        grid.addWidget(epochs, 3, 0)
        grid.addWidget(self.epochs_edit, 3, 1)
        grid.addWidget(eta, 4, 0)
        grid.addWidget(self.eta_edit, 4, 1)
        grid.addWidget(layers, 5, 0)
        grid.addWidget(self.layers_edit, 5, 1)
        params_box.addWidget(info_label)
        params_box.addLayout(grid)

        # layout for the button and hidden layers
        self.train_button.setText('Train network')
        self.train_button.clicked.connect(self.train_and_evaluate)
        self.layers_box.addWidget(self.train_button)
        self.create_layers_layout()

        # add parameters layout and layers layout to the overall layout
        general_box = QVBoxLayout()
        general_box.addLayout(params_box)
        general_box.addLayout(self.layers_box)
        self.layers_edit.currentIndexChanged.connect(self.create_layers_layout)

        self.setLayout(general_box)
        self.setGeometry(100, 100, 800, 500)
        self.setWindowIcon(QIcon('protein.png'))
        self.setWindowTitle('Secondary Structure Prediction')
        self.show()

    def create_layers_layout(self):
        layer_count = int(self.layers_edit.currentText())  # layers to be shown
        layers_on_window = len(self.layers)  # layers already shown
        # add new layout for more layers
        if layers_on_window < layer_count:
            for i in range(layers_on_window + 1, layer_count+1):
                new_layer_neurons = QComboBox()
                self.layers.append(new_layer_neurons)
                new_layer_neurons.addItems(self.nb_neurons)
                new_layer = QVBoxLayout()
                new_label = QLabel('Layer ' + str(i) + ':')
                new_layer.addWidget(new_label)
                new_layer.addWidget(new_layer_neurons)
                self.layers_box.addLayout(new_layer)
        else:
            # remove layouts for less layers
            for j in range(layers_on_window, layer_count, -1):
                layout_item = self.layers_box.itemAt(j)
                # remove all the widgets from this layout
                while layout_item.count():
                    item = layout_item.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.setParent(None)
                self.layers_box.removeItem(layout_item)
                self.layers.pop()

    def train_and_evaluate(self):
        window_size = int(self.window_edit.currentText())
        batch_size = int(self.batch_edit.currentText())
        epochs = int(self.epochs_edit.currentText())
        learning_rate = float(self.eta_edit.currentText())
        hidden_layers = []
        for layer in self.layers:
            hidden_layers.append(int(layer.currentText()))
        ann = self.controller.ann
        self.controller = Controller(ann, window_size, batch_size, epochs, learning_rate, hidden_layers)
        self.controller.train_and_evaluate()
        self.enter_protein = EvaluateProtein(self.controller)
        self.enter_protein.show()


class EvaluateProtein(QWidget):

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setFont(QFont("Arial", 18))

        box = QVBoxLayout()
        seq_label = QLabel('Enter protein sequence to predict secondary structure')
        seq_label.setWordWrap(True)
        self.sequence_input = QPlainTextEdit()
        self.sequence_input.setPlainText('ASFSEAPPGNPKAGEKIFKTKCAQCHTVDKGAGHKQGPNLNGLFGRQSGTTPGYSYSTADKNMA'+
                                         'VIWEENTLYDYLLNPKKYIPGTKMVFPGLKKPQERADLISYLKEATS')
        self.predict_button = QPushButton('Predict')
        output_label = QLabel('The predicted secondary structure for the given sequence is given below,' +
                              ' where h = alfa helix, e = beta sheet, _ = coil')
        output_label.setWordWrap(True)
        self.output = QPlainTextEdit()
        box.addWidget(seq_label)
        box.addWidget(self.sequence_input)
        box.addWidget(self.predict_button)
        box.addWidget(output_label)
        box.addWidget(self.output)
        self.setLayout(box)
        self.setGeometry(400, 400, 800, 500)
        self.setWindowIcon(QIcon('protein.png'))
        self.setWindowTitle('Secondary Structure Prediction')
        self.predict_button.clicked.connect(self.predict)
        self.show()

    def predict(self):
        sequence = self.sequence_input.toPlainText()
        for letter in sequence:
            if letter not in self.controller.encoding:
                QMessageBox.about(self, "Invalid protein sequence",
                                  "<font size=14> Please use capital one letter abbreviation for amino acids </font>")
                return
        secondary_structure = self.controller.predict_for_protein(sequence)

        self.output.setPlainText(secondary_structure)

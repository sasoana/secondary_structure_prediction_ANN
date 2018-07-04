import sys
from PyQt5.QtWidgets import QApplication

from ANN import ANN
from Controller import Controller
from UI import App

ann = ANN()
ctrl = Controller(ann)
app = QApplication(sys.argv)
UI = App(ctrl)
sys.exit(app.exec_())

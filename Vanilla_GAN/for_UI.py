import sys
import Vanilla_GAN.Vanilla_Standard as GAN
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os
import csv

class mainWidow(QWidget):
    def __init__(self):
        super().__init__()
        self.method = 0
        self.setupUI()

    def setupUI(self):
        self.setFixedSize(250, 300)
        self.setWindowTitle("GAN GUI")
        self.setWindowIcon(QIcon("logo.jpg"))

        label1 = QLabel("Batch size : ")
        label2 = QLabel("Epoch size : ")
        label3 = QLabel("Learning Rate : ")
        label4 = QLabel("Data set : ")

        self.btn1 = QPushButton("Train")
        self.btn1.clicked.connect(self.btn1_slot)
        self.btn2 = QPushButton("Terminate")
        self.btn2.clicked.connect(self.btn2_slot)
        #TODO: Test button
        self.btn3 = QPushButton("Test")
        self.btn3.setDisabled(True)

        #Batch sIze
        self.spin1 = QSpinBox()
        self.spin1.setMinimum(0)
        self.spin1.setMaximum(60000)
        self.spin1.setValue(100)

        #Epoch size
        self.spin2 = QSpinBox()
        self.spin2.setMinimum(0)
        self.spin2.setMaximum(10000)
        self.spin2.setValue(300)

        #lr
        self.spin3 = QDoubleSpinBox()
        self.spin3.setDecimals(4)
        self.spin3.setMinimum(0)
        self.spin3.setMaximum(1)
        self.spin3.setValue(0.0002)
        self.spin3.setSingleStep(0.0001)

        self.combo = QComboBox()
        self.combo.addItems(["MNIST", "Fashion MNIST", "CIFAR-10"])
        self.combo.currentIndexChanged.connect(self.method_changed)

        inner_lay = QGridLayout()
        inner_lay.addWidget(label1, 0, 0)
        inner_lay.addWidget(label2, 1, 0)
        inner_lay.addWidget(label3, 2, 0)
        inner_lay.addWidget(label4, 3, 0)

        inner_lay.addWidget(self.spin1, 0, 1)
        inner_lay.addWidget(self.spin2, 1, 1)
        inner_lay.addWidget(self.spin3, 2, 1)
        inner_lay.addWidget(self.combo, 3, 1)

        Outlay = QVBoxLayout()
        Outlay.addLayout(inner_lay)
        Outlay.addWidget(self.btn1)
        Outlay.addWidget(self.btn2)
        Outlay.addWidget(self.btn3)

        self.setLayout(Outlay)

    def method_changed(self):
        self.method =self.combo.currentIndex()

    def btn1_slot(self):
        self.Task = trainThread(self.method, self.spin1.value(), self.spin2.value(), self.spin3.value())
        self.Task.start()

    def btn2_slot(self):
        if self.Task.isRunning():
            self.Task.f.close()
            self.Task.terminate()
        print("Terminated")

class trainThread(QThread):
    taskFinished = pyqtSignal()
    def __init__(self, method, batch_sz, epoch_sz, lr):
        super().__init__()
        self.method = method
        self.batch_sz = batch_sz
        self.lr = lr
        self.epoch_sz = epoch_sz
        self.f, self.write = self.log_write()

    def run(self):
        GAN.train(self.method, self.batch_sz, self.epoch_sz, self.lr, self.write)
        self.taskFinished.emit()
        self.f.close()

    def log_write(self):
        #MNIST : 0, Fashion_MNIST : 1, CIFAR-10 : 2
        if self.method ==0:
            result_dir = './MNIST_Results'
        elif self.method == 1:
                result_dir = './FashionMNIST_Results'
        elif self.method == 2:
                result_dir = './CIFAR10_Results'
        # check the directory
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        f = open(os.path.join(result_dir) + '/loss_log.csv', 'w', newline='')
        write = csv.writer(f)
        write.writerow(['Epoch', 'Step', 'd_loss', 'g_loss', 'D(x)', 'D(G(z))'])

        return f, write

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainWidow()
    window.show()
    app.exec_()
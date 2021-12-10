import numpy as np

class NeuralNetwork:
    def __init__(self, width):
        self.learningRate = 0.1
        self.d = 0.1
        self.epoch = 100
        self.gamma = 0.1
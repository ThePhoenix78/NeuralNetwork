import numpy as np


class Layer():
    """
    Layer of the Neural Network
    Parameters:
        input_size (int) : the size of the input layer
        output_size (int): the size of the output layer
        weight (list)    : the list of weights
        bias (list)      : the list of bias
    """

    def __init__(self, input_size: int = None,
                 output_size: int = None,
                 activation_function: str = "sigmoid",
                 weight: list = None,
                 bias: list = None,
                 dense: bool = False):

        self.bias = None

        if input_size and output_size:
            self.weight = np.random.randn(input_size, output_size)

        elif weight:
            self.weight = np.asarray(weight)

        if bias:
            self.bias = np.asarray(bias)

        self.select_activation_function(activation_function.lower())

        self.dense = dense
        self.input_size = len(self.weight)
        self.output_size = len(self.weight[0])

    def reset(self):
        self.bias = None
        self.weight = np.random.randn(self.input_size, self.output_size)

    def select_activation_function(self, type: str):
        self.type = type

        if self.type == "sigmoid":
            self.act = self.sigmoid
            self.der = self.sigmoid_prime

        elif self.type == "relu":
            self.act = self.relu
            self.der = self.relu_prime

        elif self.type == "tanh":
            self.act = self.tanh
            self.der = self.tanh_prime

        elif self.type == "swish":
            self.act = self.swish
            self.der = self.swish_prime

    def sigmoid(self, s, b: int = 1):
        return 1 / (1 + np.exp(-b*s))
        # return .5 * (1 + np.tanh(.5 * s))

    def sigmoid_prime(self, s):
        return s * (1 - s)

    def swish(self, x, b: int = 1):
        return x * self.sigmoid(x, b)

    def swish_prime(self, x, b: int = 1):
        return (b * self.swish(x, b)) + (self.sigmoid(x, b) * (1 - (b * self.swish(x, b))))

    def relu(self, s):
        return np.maximum(0, s)

    def relu_prime(self, s):
        for i in range(len(s)):
            for j in range(len(s[0])):
                s[i][j] = 0 if s[i][j] < 0 else 1
        return s

    def tanh(self, s):
        return np.tanh(s)

    def tanh_prime(self, s):
        return 1-s**2

    def activate(self, s):
        self.bias = self.act(s)

    def derivate(self, o=None):
        if isinstance(o, type(None)):
            o = self.bias
        return self.der(o)

    def predict(self, s):
        return self.act(s)

    def build_str(self):
        res = ""
        for key, value in self.__dict__.items():
            res += f"{key} : {value}\n"

        return res

    def __str__(self):
        return self.build_str()

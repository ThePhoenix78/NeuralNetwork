from random import randint, random, shuffle, choice
import numpy as np
import json

try:
    from .Mutation import *
    from .numpy_encoder import *
except ImportError:
    from Mutation import *
    from numpy_encoder import *


class Layer():
    """
    Layer of the Neural Network
    Parameters:
        input_size (int)          : the size of the input layer
        output_size (int)         : the size of the output layer
        activation_function (str) : the activation activation_function
        weight (list)             : the list of weights
        bias (list)               : the list of bias
    """

    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 activation_function: str = "sigmoid",
                 weight: list = None,
                 bias: list = None,
                 dense: bool = True
            ):

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
        self.activation_function = type

        if self.activation_function == "sigmoid":
            self.act = self.sigmoid
            self.der = self.sigmoid_prime

        elif self.activation_function == "relu":
            self.act = self.relu
            self.der = self.relu_prime

        elif self.activation_function == "tanh":
            self.act = self.tanh
            self.der = self.tanh_prime

        elif self.activation_function == "swish":
            self.act = self.swish
            self.der = self.swish_prime

        else:
            raise "Error! Invalid activation function (sigmoid, relu, tanh, swish)"

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


class Layers(Mutations):
    def __init__(self,
            layers: list = [],
        ):
        Mutations.__init__(self)
        self.layers = layers
        self.size = len(self.layers)

    def save(self, name: str = "neural.json"):
        dico = {
            "weight": [self.layers[i].weight for i in range(self.size)],
            "bias": [self.layers[i].bias for i in range(self.size)],
            "activation_function": [self.layers[i].activation_function for i in range(self.size)]
        }

        with open(name, "w", encoding="utf8") as f:
            f.write(json.dumps(dico, indent=4, cls=NumpyEncoder))

    def load(self, trained_set: str = "neural.json"):
        with open(trained_set, "r", encoding="utf8") as f:
            return json.load(f)

    def get_layers_size_metadata(self):
        metadata = []
        for i in range(1, len(self.layers)):
            metadata.append(self.layers[i].input_size)

        return metadata

    def link_layers(self, input_size, output_size):
        current = self.layers[0]

        self.layers[0].input_size = input_size

        for i in range(1, len(self.layers)):
            self.layers[i].input_site = current.output_size
            current = self.layers[i]

        self.layers[-1].output_size = output_size

    def reverse_link_layers(self, input_size, output_size):
        current = self.layers[-1]
        i = len(self.layers)-2

        self.layers[-1].output_size = output_size

        while i > 0:
            self.layers[i].output_size = current.input_size
            current = self.layers[i]
            i -= 1

        self.layers[0].input_size = input_size

    def reset_neural_network(self):
        """
        will reset all the weights and bias of the neural network
        """
        for i in range(len(self.layers)):
            self.layers[i].reset()

    def mutate_layers(self,
                      mutation_chance: int = 50,
                      current_precision: int = 50,
                    ):
        mutate = True if mutation_chance > randint(1, 100) else False

        if mutate:
            pass

    def generate_layers(self):
        layer = []

        for i in range(len(self.layers)):
            layer.append(Layer(self.layers[i].input_size, self.layers[i].output_size, self.layers[i].activation_function))

        return layer

    def add_layer(self,
                  input_size: int = None,
                  output_size: int = None,
                  activation_function: str = "sigmoid",
                  weight: list = None,
                  bias: list = None,
                  dense: bool = True
            ):
        self.layers.append(Layer(input_size=input_size, output_size=output_size, weight=weight, bias=bias, activation_function=activation_function, dense=dense))
        self.size = len(self.layers)

    def put_layer(self, layer: Layer):
        self.layers.append(layer)
        self.size = len(self.layers)

    def forward(self, data):
        """
        forward method of the Neural Network
        """
        z = np.dot(data, self.layers[0].weight)
        self.layers[0].activate(z)

        for i in range(1, self.size):
            z1 = np.dot(self.layers[i-1].bias, self.layers[i].weight)
            self.layers[i].activate(z1)

    def backward(self, data, output, learning_rate: int = 1):
        """
        backward method of the Neural Network
        """

        out = self.layers[-1].bias

        out_error = output - out
        out_delta = out_error * self.layers[-1].derivate(out)

        z_delta = []
        t_delta = out_delta

        for i in range(1, self.size):
            z1_error = t_delta.dot(self.layers[-i].weight.T)
            z1_delta = z1_error * self.layers[-i-1].derivate()
            z_delta.append(z1_delta)
            t_delta = z1_delta

        self.layers[0].weight += data.T.dot(z_delta[-1]) * learning_rate

        for i in range(self.size-2):
            self.layers[i+1].weight += self.layers[i].bias.T.dot(z_delta[-i-2]) * learning_rate

        self.layers[-1].weight += self.layers[-2].bias.T.dot(out_delta) * learning_rate


if __name__ == "__main__":
    layers = Layers([
        Layer(5, 10, "sigmoid"),
        Layer(10, 10, "sigmoid")
    ])
    print(layers.get_layers_size_metadata())

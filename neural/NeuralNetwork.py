from .to_table import make_table
from .Flatten import flatten
from .Layer import Layer
from .numpy_encoder import NumpyEncoder

import numpy as np
from copy import deepcopy
from random import randint, random, shuffle, choice
import json
import warnings


warnings.filterwarnings('ignore')


class NeuralNetwork(object):
    def __init__(self, inputs: list,
                 results: list,
                 training_set: list = None,
                 hidden_layers: list = [[randint(3, 9), "sigmoid"]],
                 activation_function: str = "sigmoid",
                 trained_set: str = None,
                 ):

        self._inputs = np.array(inputs, dtype=float)

        if not isinstance(results[0], (list, tuple)):
            results = [[elem] for elem in results]

        self.output = np.array(results, dtype=float)

        self.training_set = training_set

        if not training_set:
            self.select_training_set()

        self.data = self._inputs/np.amax(self._inputs, axis=0)

        self.layers = []

        self.activation_function = activation_function
        self.input_size = len(self.data[0])
        self.output_size = len(self.output[0])
        self.hidden_layers = []
        self.size = 0

        if trained_set:
            data = self.load(trained_set)

            for i in range(len(data["type"])):
                self.set_layer(data["weight"][i], data["bias"][i], data["type"][i])

            self.size = len(self.layers)
            return

        for i in range(len(hidden_layers)):
            if isinstance(hidden_layers[i], int):
                hidden_layers[i] = [hidden_layers[i], "sigmoid"]

            hidden_layers[i] = list(hidden_layers[i])

            if hidden_layers[i][1] not in ["sigmoid", "relu", "tanh", "swish"]:
                hidden_layers[i][1] = "sigmoid"

        self.hidden_layers = hidden_layers

        self.reset_neural_network()

    def reset_neural_network(self):
        """
        will reset all the weights and bias of the neural network
        """
        self.layers = []

        self.size = len(self.hidden_layers)+1

        self.input_size = len(self.data[0])
        self.output_size = len(self.output[0])

        # self.add_layer(input_size, self.hidden_layers[0][0], self.activation_function)
        self.add_layer(self.input_size, self.hidden_layers[0][0], self.hidden_layers[0][1])

        for i in range(self.size-2):
            self.add_layer(self.hidden_layers[i][0], self.hidden_layers[i+1][0], self.hidden_layers[i+1][1])

        # self.add_layer(self.hidden_layers[-1][0], output_size, self.hidden_layers[-1][1])
        self.add_layer(self.hidden_layers[-1][0], self.output_size, self.activation_function)

        # self.size = len(self.layers)

    def reset_and_shuffle_neural_network(self):
        for i in range(len(self.hidden_layers)):
            val = self.hidden_layers[i][0]
            self.hidden_layers[i][0] = randint(val-val//2, val+val//2)

        self.reset_neural_network()

    def generate_scheme(self):
        layer = []
        # layer.append(Layer((input_size, self.hidden_layers[0][0], self.activation_function))
        layer.append(Layer(self.input_size, self.hidden_layers[0][0], self.hidden_layers[0][1]))

        for i in range(len(self.hidden_layers)-1):
            layer.append(Layer(self.hidden_layers[i][0], self.hidden_layers[i+1][0], self.hidden_layers[i+1][1]))

        # layer.append(Layer(self.hidden_layers[-1][0], output_size, self.hidden_layers[-1][1]))
        layer.append(Layer(self.hidden_layers[-1][0], self.output_size, self.activation_function))
        return layer

    def save(self, name: str = "Neural.json"):
        dico = {
            "weight": [self.layers[i].weight for i in range(self.size)],
            "bias": [self.layers[i].bias for i in range(self.size)],
            "type": [self.layers[i].type for i in range(self.size)]
        }

        with open(name, "w") as f:
            f.write(json.dumps(dico, indent=4, cls=NumpyEncoder))

    def load(self, trained_set: str = "Neural.json"):
        with open(trained_set, "r") as f:
            return json.load(f)

    def add_layer(self, input_size, output_size, activation_function: str = "sigmoid", dense: bool = False):
        self.layers.append(Layer(input_size=input_size, output_size=output_size, activation_function=activation_function, dense=dense))

    def set_layer(self, weight, bias, activation_function):
        self.layers.append(Layer(weight=weight, bias=bias, activation_function=activation_function))

    def put_layer(self, layer: Layer):
        self.layers.append(layer)

    def select_training_set(self):
        """
        will select a training set from the current dataset
        """
        self.training_set = []
        tests = []
        results = []

        sets = []

        for k, v in zip(self._inputs, self.output):
            sets.append([k, v])

        shuffle(sets)
        to_delete = []

        for i in range(len(sets)):
            if sets[i][1].tolist() not in results:
                tests.append(sets[i][0])
                results.append(sets[i][1].tolist())
                to_delete.append(i)

        np.delete(self._inputs, to_delete)
        np.delete(self.output, to_delete)

        for k, v in zip(tests, results):
            self.training_set.append([k, v])

    def cut_dataset(self, dataset: list, divide: int = None):
        """
        can divide a dataset in multiple smaller datasets

        args:
            dataset     : the dataset
            divide      : the amount of sub-dataset

        return:
            a list of dataset
        """
        if divide in (0, 1) or not divide:
            return [dataset]

        size = len(dataset)
        cut0 = size//divide

        t_data = []
        c = 0
        cut = cut0
        for n in range(1, divide+1):
            t_data.append(dataset[c:cut])
            c = cut
            cut = cut0 * (n+1)

        if c < size:
            t_data.append(dataset[c:])

        return t_data

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

    def deep_train(self,
                   epoch: int = 1000,
                   learning_rate: int = 1,
                   error: int = 0,
                   divide_set: int = None
                   ):
        """
        deep learning
        use forward and backward propagation to learn from it's mistake

        args:
            epoch           : the number of iterations
            error           :
            learning_rate   : adjust the precision
            divide_set      : chances of neuron network mutation

        return:
            nothing
        """
        data = self.cut_dataset(self.data, divide_set)
        output = self.cut_dataset(self.output, divide_set)

        for k in range(epoch):
            for i in range(len(data)):
                self.forward(data[i])
                self.backward(data[i], output[i], learning_rate)

            if error and self.compare_single() <= error:
                break

    def smart_train(self,
                    learning_method: str = "deep",
                    epoch: int = 100,
                    learning_rate: int = "random",
                    population_size: int = "random",
                    genetic_mutation: int = 40,
                    cool: list = 300,
                    max_retry: int = 10,
                    reset: bool = True,
                    mutation: list = None,
                    error: int = 5,
                    divide_set: int = 1
                    ):
        """
        ❗ experimental method ❗
        """

        if isinstance(mutation, int):
            mutation = [mutation, ["sigmoid"]]

        reset2 = True if mutation else False
        j = 0
        k = 0

        t_size = len(self.training_set)
        res = t_size

        while not (((res*100)/t_size) <= error) and k < max_retry:
            if learning_method == "deep":
                a = learning_rate

                if learning_rate == "random":
                    a = random()

                self.deep_train(epoch=epoch, learning_rate=a, error=error, divide_set=divide_set)

            elif learning_method == "genetic":
                a = population_size
                if population_size == "random":
                    a = randint(5, 100)

                self.genetic_train(population_size=a, epoch=epoch, init_layer=self.layers, error=error, mutation=genetic_mutation)

            res = 0
            for i in range(t_size):
                val = self.predict(self.training_set[i][0], True)

                a = (val == self.training_set[i][1])
                if False in a:
                    res += 1

                if ((res*100)/t_size) > error:
                    break

            if j == cool:
                j = 0
                k += 1
                self.reset_neural_network()

            if k == max_retry//2 and reset:
                reset = False
                self.reset_and_shuffle_neural_network()

            if k == max_retry//1.5 and mutation and reset2:
                reset2 = False
                a = randint(1, 100)
                if a >= 100-mutation[0]:
                    self.hidden_layers.append([randint(3, 25), choice(mutation[1])])

                elif a <= mutation[0] and len(self.hidden_layers) > 1:
                    self.hidden_layers.pop()

            j += 1

        if k >= max_retry:
            return False

        return True

    def special_train(self,
                      learning_method: str = "shuffle",
                      epoch: int = 100,
                      learning_rate: int = "random",
                      population_size: int = "random",
                      cool: int = 3,
                      max_retry: int = 10,
                      reset: bool = True,
                      mutation: int = None,
                      error: int = 15,
                      divide_set: int = 1,
                      absolute_end: int = 100
                      ):
        """
        ❗ experimental method ❗
        """
        val = False
        i = 0
        learning = learning_method

        select = 0

        while not val and i < absolute_end:
            if learning_method == "shuffle":
                learning = choice(["genetic", "deep"])

            elif isinstance(learning_method, (list, tuple)):
                learning = learning_method[select]
                select += 1

                if select > len(learning_method):
                    select = 0

            print(f"Iter : {i+1}")

            val = self.smart_train(learning_method=learning,
                                   epoch=epoch,
                                   learning_rate=learning_rate,
                                   cool=cool,
                                   max_retry=max_retry,
                                   reset=reset,
                                   mutation=mutation,
                                   error=error,
                                   divide_set=divide_set
                                   )
            i += 1

        return val

    def mix_genome(self, give: Layer, receive: Layer, threshold: int = 50):
        if give.input_size != receive.input_size or give.output_size != receive.output_size:
            raise "Error! Layers aren't the same!"

        for i in range(give.input_size):
            a = randint(1, 100)
            if a >= 100-threshold:
                receive.weight[i] = give.weight[i]

    def mix_adn(self, give: Layer, receive: Layer, threshold: int = 50):
        if give.input_size != receive.input_size or give.output_size != receive.output_size:
            raise "Error! Layers aren't the same!"

        for i in range(give.input_size):
            a = randint(1, 100)
            if a >= 100-threshold:
                for j in range(give.output_size):
                    b = randint(1, 100)
                    if b >= 100-threshold:
                        receive.weight[i][j] = give.weight[i][j]

    def genetic_train(self,
                      population_size: int = 10,
                      epoch: int = 10,
                      error: int = 0,
                      threshold_error: int = 0,
                      mutation: int = 50,
                      method: int = 1,
                      deep: bool = False,
                      init_layer: Layer = None
                      ):
        """
        genetic learning
        will take the best of each epoch and reproduce it with the new generation

        args:
            population_size : size of the population
            epoch           : the number of iterations
            error           :
            threshold_error : the reliability threshold
            mutation        : chances of neuron network mutation
            method          : 1 = mix only the layer 2 = mix all the weights
            deep            : allows you to adjust with deep learning (experimental)
            init_layer      : the init layer (not required)

        return:
            the result layer
        """
        layers = [self.generate_scheme() for i in range(population_size)]

        if isinstance(init_layer, Layer):
            layers.append(deepcopy(init_layer))
            layers.pop(0)

        comp_sing = [0 for i in range(population_size)]
        # comp_all = [0 for i in range(population_size)]
        sup = False

        for e in range(epoch):
            for i in range(population_size):
                # ind[i] = self.predict(prediction=self.data, round=False, layers=layers[i])
                # self.show_result(True, "all", layers[i])
                # comp_all[i] = self.compare_all(layers=layers[i])

                comp_sing[i] = self.compare_single(layers=layers[i])

            err = min(comp_sing)

            id_sing = comp_sing.index(err)

            self.layers = layers[id_sing]

            if err <= threshold_error:
                sup = True
                break

            next_gen = [self.generate_scheme() for i in range(population_size-1)]

            for j in range(len(next_gen)):
                for k in range(len(next_gen[j])):
                    if method == 1:
                        self.mix_genome(give=self.layers[k], receive=next_gen[j][k], threshold=mutation)
                    elif method == 2:
                        self.mix_adn(give=self.layers[k], receive=next_gen[j][k], threshold=mutation)

            next_gen.append(self.layers)
            layers = next_gen

        if sup and deep:
            prime = deepcopy(self.layers)
            self.show_result(True, "all")
            print("training...")

            self.deep_train(100, 1, error)

            p1 = self.compare_single(prime)
            p2 = self.compare_single(self.layers)

            if p1 < p2:
                self.layers = prime

        return self.layers

    def compare_all(self, layers: list = None):
        """
        compare all the elements from the training set with the actual result
        """
        res = []
        res2 = []
        for i in range(len(self.training_set)):
            res.append(self.predict(prediction=self.training_set[i][0], round=True, layers=layers))
            res2.append(self.training_set[i][1])

        res = np.array(res)
        res2 = np.array(res2)
        val = (res == res2)

        rel = 0

        size = res.size * res[0][1].size

        for elem in val:
            rel += sum([1 for e in elem if not e])

        return rel * 100 / size

    def compare_single(self, layers: list = None):
        """
        compare all the results from the training set with the actual result
        """
        res = 0

        for i in range(len(self.training_set)):
            val = self.predict(prediction=self.training_set[i][0], round=True, layers=layers)

            a = (val == self.training_set[i][1])
            if False in a:
                res += 1

        return res * 100 / len(self.training_set)

    def show_result(self,
                    round: bool = True,
                    maximum: int = "all",
                    layers: list = None,
                    show_layer_info: bool = True
                    ):
        """
        will display the results of the training in a table

        args:
            round           : if the results need to be rounded (0 or 1)
            maximum         : the maximum elements displayed
            layers          : the layer to compare and show results
            show_layer_info : display the layer's inforamtions

        return:
            nothing
        """

        if maximum == "all":
            maximum = len(self.training_set)

        res = []
        a = len(self.training_set)
        maxi = min(a, maximum)

        count_error = 0
        for i in range(maxi):
            val = self.predict(prediction=self.training_set[i][0], round=round, layers=layers)
            same = True
            a = (val == self.training_set[i][1])

            if False in a:
                same = False

            nb_error = 0
            for nb in a:
                if not nb:
                    nb_error += 1
                    count_error += 1

            res.append([val, self.training_set[i][1], same, nb_error])

        size = len(self.training_set) * len(self.training_set[0][1])

        print("-"*50)
        if show_layer_info:
            if not layers:
                lay = self.layers
            print(f"Layers : {[lay[i].input_size for i in range(1, len(lay))]} | input size {self.input_size} | output size : {self.output_size}")
        print(f"Errors      : {self.compare_all(layers=layers):.2f}% ({count_error}/{size} errors)")
        print(f"Error lines : {self.compare_single(layers=layers):.2f}%")
        print(make_table(labels=["Neural Network", "Test", "Equals", "Nb error"], rows=res, left=["Index"]+[i+1 for i in range(maxi)], centered=True))
        print("-"*50)

    def predict(self, prediction, round: bool = False, layers: list = None):
        """
        will predict a result based on the current neural network

        args:
            prediction  : the matrix to test
            round       : if the results need to be rounded (0 or 1)
            layers      : the layer to compare and show results

        return:
            the result matrix
        """

        if not layers:
            layers = self.layers

        res = [0] * self.size
        z = np.dot(prediction, layers[0].weight)
        res[0] = layers[0].predict(z)

        for i in range(1, self.size):
            z1 = np.dot(res[i-1], layers[i].weight)
            res[i] = layers[i].predict(z1)

        result = res[-1]

        if round:
            return np.matrix.round(result, 0)

        return result

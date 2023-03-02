from .to_table import make_table
# from .Flatten import flatten
from .Layer import *
from .numpy_encoder import NumpyEncoder

import numpy as np
from copy import deepcopy
from random import randint, random, shuffle, choice
import json
import warnings


warnings.filterwarnings('ignore')


class NeuralNetwork(Layers):
    def __init__(self,
                 inputs: list,
                 results: list,
                 training_set: list = None,
                 layers: Layers = [],
                 trained_set: str = None,
                 manual_entry: bool = False
                 ):

        if isinstance(layers, Layers):
            layers = layers.layers

        Layers.__init__(self, layers)

        self._inputs = np.array(inputs, dtype=float)

        if not isinstance(results[0], (list, tuple)):
            results = [[elem] for elem in results]

        self.output = np.array(results, dtype=float)

        self.training_set = training_set

        if not training_set:
            self.select_training_set()

        self.data = self._inputs/np.amax(self._inputs, axis=0)

        self.input_size = len(self.data[0])
        self.output_size = len(self.output[0])
        # self.reverse_link_layers(self.input_size, self.output_size)

        if not manual_entry:
            self.add_layer(index=0, layer=Layer(self.input_size, self.layers[0].input_size))
            self.add_layer(layer=Layer(self.layers[-1].output_size, self.output_size))

        # activations_function =  ["sigmoid", "relu", "tanh", "swish"]

        if trained_set:
            self.layers = Layers()
            data = self.load(trained_set)

            for i in range(len(data["activation_function"])):
                self.add_layer(weight=data["weight"][i], bias=data["bias"][i], activation_function=data["activation_function"][i])

            self.size = len(data["activation_function"])
            return

        self.reset_neural_network()

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

            if error and self.compare() <= error:
                break

    def smart_train(self,
                    learning_method: str = "genetic",
                    epoch: int = 100,
                    learning_rate: int = 1,
                    population_size: int = 25,
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
                self.deep_train(epoch=epoch, learning_rate=learning_rate, error=error, divide_set=divide_set)

            elif learning_method == "genetic":
                self.genetic_train(population_size=population_size, epoch=epoch, init_layer=self.layers, error=error, mutation=genetic_mutation)

            res = 0
            for i in range(t_size):
                val = self.predict(self.training_set[i][0], True)

                if False in (val == self.training_set[i][1]):
                    res += 1

                if ((res*100)//t_size) > error:
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

                if a >= 100-mutation[0]//2:
                    index = randint(1, self.size-1)
                    self.add_layer(index=index, layer=Layer(input_size=a, output_size=a, activation_function=choice(mutation[1])))
                    self.link_layers()

                elif a <= mutation[0]//2 and self.size > 2:
                    self.pop_layer()
                    self.link_layers()

            j += 1

        if k >= max_retry:
            return False

        return True

    def special_train(self,
                      learning_method: str = "genetic",
                      epoch: int = 100,
                      learning_rate: int = 1,
                      population_size: int = 25,
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
            if isinstance(learning_method, (list, tuple)):
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
                                   divide_set=divide_set)
            i += 1

        return val

    def mix_genome(self, give: Layer, receive: Layer, threshold: int = 50):
        if give.input_size != receive.input_size or give.output_size != receive.output_size:
            raise "Error! Layers aren't the same!"

        for i in range(give.input_size):
            if randint(1, 100) >= 100-threshold:
                receive.weight[i] = give.weight[i]

    def mix_adn(self, give: Layer, receive: Layer, threshold: int = 50):
        if give.input_size != receive.input_size or give.output_size != receive.output_size:
            raise "Error! Layers aren't the same!"

        for i in range(give.input_size):
            if randint(1, 100) >= 100-threshold:
                for j in range(give.output_size):
                    if randint(1, 100) >= 100-threshold:
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
        layers = [self.generate_layers() for i in range(population_size)]

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

                comp_sing[i] = self.compare(layers=layers[i])

            err = min(comp_sing)

            id_sing = comp_sing.index(err)

            self.layers = layers[id_sing]

            if err <= threshold_error:
                sup = True
                break

            next_gen = [self.generate_layers() for i in range(population_size-1)]

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

            p1 = self.compare(prime)
            p2 = self.compare(self.layers)

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

    def compare(self, layers: list = None):
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
        count_error2 = 0
        for i in range(maxi):
            val = self.predict(prediction=self.training_set[i][0], round=round, layers=layers)
            same = True
            a = (val == self.training_set[i][1])

            if False in a:
                count_error2 += 1
                same = False

            nb_error = 0
            for nb in a:
                if not nb:
                    nb_error += 1
                    count_error += 1

            res.append([val, self.training_set[i][1], same, nb_error])

        size = len(self.training_set) * len(self.training_set[0][1])
        size2 = len(self.training_set)

        print("-"*50)
        if show_layer_info:
            if not layers:
                lay = self.layers

            print(f"Layers : {[[lay[i].input_size, lay[i].activation_function] for i in range(1, len(lay))]} | input size {self.input_size} | output size : {self.output_size}")
        print(f"Errors      : {self.compare_all(layers=layers):.2f}% ({count_error}/{size} errors)")
        print(f"Error lines : {self.compare(layers=layers):.2f}% ({count_error2}/{size2} errors)")
        print(make_table(labels=["Neural Network", "Training Set", "Equals", "Nb error"], rows=res, left=["Index"]+[i+1 for i in range(maxi)], centered=True))
        print("-"*50)

    def predict(self, prediction: list, round: bool = True, layers: list = None):
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

        res = [0] * len(layers)
        z = np.dot(prediction, layers[0].weight)
        res[0] = layers[0].predict(z)

        for i in range(1, len(layers)):
            z1 = np.dot(res[i-1], layers[i].weight)
            res[i] = layers[i].predict(z1)

        result = res[-1]

        if round:
            return np.matrix.round(result, 0)

        return result

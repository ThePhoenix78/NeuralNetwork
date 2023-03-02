from random import randint, random, shuffle, choice
import numpy as np
import json


class Mutation:
    def __init__(self, precision: int = 50, layers: list = [], generation: int = 0):
        self.generation = generation
        self.precision = precision
        self.layers = layers
        self.size = len(layers)

    def build_str(self):
        res = ""
        for key, value in self.__dict__.items():
            res += f"{key} : {value}\n"

        return res

    def __str__(self):
        return self.build_str()


class Mutations:
    def __init__(self, mutation: Mutation = []):
        self.mutations = mutation

        if mutation:
            self.mutations = [mutation]

        self.lenght = len(self.mutations)

    def put_mutation(self, mutation: Mutation):
        self.mutations.append(mutation)
        self.lenght = len(self.mutations)
        self.index_mutations()

    def add_mutation(self, precision: int, layers: list, generation: int = 0):
        self.mutations.append(Mutation(generation=generation, precision=precision, layers=layers))
        self.lenght = len(self.mutations)
        self.index_mutations()

    def index_mutations(self):
        for i in range(len(self.mutations)):
            self.mutations[i].generation = i

    def get_pente_precision(self):
        pente = 0
        current = self.mutations[0]

        for i in range(1, len(self.mutations)):
            pente += (self.mutations[i].precision - current.precision) * i
            current = self.mutations[i]

        return pente

    def get_pente_size(self):
        pente = 0
        current = self.mutations[0]

        for i in range(1, len(self.mutations)):
            pente += (self.mutations[i].size - current.size) * i
            current = self.mutations[i]

        return pente

    def get_min_max(self):
        current_max = self.mutations[0]
        current_min = self.mutations[0]

        for i in range(1, len(self.mutations)):
            if self.mutations[i].precision > current_max.precision:
                current_max = self.mutations[i]

            if self.mutations[i].precision < current_min.precision:
                current_min = self.mutations[i]

        return current_max, current_min


    def calc_mutations(self, current_precision):
        pass


if __name__ == "__main__":
    mutations = Mutations(Mutation(50, [15, 3, 15]))
    mutations.add_mutation(55, [15, 7, 14])
    mutations.add_mutation(58, [15, 15])
    mutations.add_mutation(80, [15, 12, 13, 14])
    print(mutations.get_pente_precision())
    print(mutations.get_pente_size())
    a, b = mutations.get_min_max()
    print(a, "\n", b)

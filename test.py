import os
# import random
# import matplotlib as plt
# import numpy as np
from neural import NeuralNetwork, flatten

size = 28

images_path = "images"

# image set
parapluie_path = f"{images_path}/parapluie"
ballon_path = f"{images_path}/ballon"
cacahuete_path = f"{images_path}/cacahuete"
bp_path = f"{images_path}/bp"

# testing set
test_path = f"{images_path}/test"


# size to flatten
x = 9

name = []
test = []
testy = []

entre = []
sort = []

for im in os.listdir(parapluie_path):
    entre.append(flatten(f"{parapluie_path}/{im}", x))
    sort.append([1, 0, 0, 0])

for im in os.listdir(ballon_path):
    entre.append(flatten(f"{ballon_path}/{im}", x))
    sort.append([0, 1, 0, 0])

for im in os.listdir(cacahuete_path):
    entre.append(flatten(f"{cacahuete_path}/{im}", x))
    sort.append([0, 0, 1, 0])

for im in os.listdir(bp_path):
    entre.append(flatten(f"{bp_path}/{im}", x))
    sort.append([0, 0, 0, 1])


training_set = []
ignore = ["at", "ct"]


for im in os.listdir(test_path):
    ign = False
    for e in ignore:
        if e in im:
            ign = True
            break

    if ign:
        continue

    name.append(im)

    if "bt" in im:
        training_set.append([flatten(f"{test_path}/{im}", x), [0, 1, 0, 0]])
    elif "bpt" in im:
        training_set.append([flatten(f"{test_path}/{im}", x), [0, 0, 0, 1]])
    elif "pt" in im:
        training_set.append([flatten(f"{test_path}/{im}", x), [1, 0, 0, 0]])
    elif "ct" in im:
        training_set.append([flatten(f"{test_path}/{im}", x), [0, 0, 1, 0]])
    else:
        training_set.append([flatten(f"{test_path}/{im}", x), [0, 0, 0, 0]])

# 2 hiddens layers (25 and 10 length)
hidden_layers = [
     # [256, "sigmoid"],
     [25, "sigmoid"],
     [10, "sigmoid"]
]
NN = NeuralNetwork(inputs=entre, results=sort, training_set=training_set, hidden_layers=hidden_layers, activation_function="sigmoid")  # , trained_set="models/123.json")

print("Starting training...")
a = NN.deep_train(learning_method="genetic", population_size=50, learning_rate=1, cool=30, reset=True, max_retry=50, epoch=10, mutation=0, error=5, divide_set=0, absolute_end=5)
# NN.genetic_train(population_size=10, epoch=50, error=0, threshold_error=0, mutation=60, method=1)
# NN.train(epoch=100, learning_rate=2, error=10, divide_set=50)
# NN.genetic_train(100, 1000, error=10, threshold_error=15, mutation=70, method=1, deep=True)

# print(NN.predict(training_set[0][0], True))
NN.show_result(True, 1500)


if input("save? (y/n) : ") == "y":
    name = input("model name? : ")
    if not name.endswith(".json"):
        name += ".json"

    NN.save(f"models/{name}")

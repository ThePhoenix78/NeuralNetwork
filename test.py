import os
from neural import NeuralNetwork, flatten


# dataset
images_path = "images"
parapluie_path = f"{images_path}/parapluie"
ballon_path = f"{images_path}/ballon"
cacahuete_path = f"{images_path}/cacahuete"
bp_path = f"{images_path}/bp"

# testing set
test_path = f"{images_path}/test"


# size to flatten
x = 7

name = []
test = []
testy = []

entre = []
sortie = []

for im in os.listdir(parapluie_path):
    entre.append(flatten(f"{parapluie_path}/{im}", x))
    sortie.append([1, 0, 0, 0])

for im in os.listdir(ballon_path):
    entre.append(flatten(f"{ballon_path}/{im}", x))
    sortie.append([0, 1, 0, 0])

for im in os.listdir(cacahuete_path):
    entre.append(flatten(f"{cacahuete_path}/{im}", x))
    sortie.append([0, 0, 1, 0])

for im in os.listdir(bp_path):
    entre.append(flatten(f"{bp_path}/{im}", x))
    sortie.append([0, 0, 0, 1])


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
     [25, "sigmoid"],
     [10, "sigmoid"]
]

NN = NeuralNetwork(inputs=entre, results=sortie, training_set=training_set, hidden_layers=hidden_layers, activation_function="sigmoid")

NN.show_result()

print("Starting training...")

method = "deep"
epoch = 50
if method == "deep":
    epoch = 500

a = NN.special_train(learning_method=method, population_size=25, learning_rate=1, cool=10, reset=True, max_retry=10, epoch=epoch, mutation=50, error=10, absolute_end=5)
# a = NN.smart_train(learning_method=method, population_size=25, learning_rate=1, cool=10, reset=True, max_retry=10, epoch=epoch, mutation=40, error=10)
# NN.genetic_train(population_size=50, epoch=150, error=10, threshold_error=0, mutation=40, method=2)
# NN.deep_train(epoch=1500, learning_rate=1, error=10)


NN.show_result()


if input("save? (y/n) : ") == "y":
    name = input("model name? : ")
    if not name.endswith(".json"):
        name += ".json"

    NN.save(f"{name}")

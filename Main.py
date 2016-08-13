from NeuralNetwork import NeuralNetwork
from Trainer import Trainer
import FeatureLoader
import numpy as np

X, Y = FeatureLoader.get_training_data()
NN = NeuralNetwork(X.shape[1]-1, 20, 8)
trainer = Trainer(NN)
trainer.train(X, Y)

times = int(input())
i = 0
print(X.shape[1]-1)
while i < times:
    NN.forward(FeatureLoader.get_testing_data(input().lower()))
    print((np.argmax(NN.a3,axis=1)+1)[0, 0])
    i += 1

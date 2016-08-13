import numpy as np


class NeuralNetwork:
    # Constructor
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.Theta1 = NeuralNetwork.random_initialize(hidden_layer_size, input_layer_size)
        self.Theta2 = NeuralNetwork.random_initialize(output_layer_size, hidden_layer_size)

    # Forward Neural Network
    def forward(self, x):
        self.z2 = np.dot(x, self.Theta1.transpose())
        self.a2 = NeuralNetwork.sigmoid(self.z2)
        self.a2 = np.hstack((np.ones(self.z2.shape[0]).reshape(self.z2.shape[0], 1), self.a2))
        self.z3 = np.dot(self.a2, self.Theta2.transpose())
        # Compute h(x)
        self.a3 = NeuralNetwork.sigmoid(self.z3)

    def cost_function(self, x, y):
        # Return sum(-y.*log(a3)-(1-y).*log(1-a3))/m
        return np.sum(np.multiply(-y, np.log(self.a3))-np.multiply((1-y), np.log(1-self.a3)))/x.shape[0]

    def costs(self, x, y):
        # Return J, grad
        self.forward(x)
        return self.cost_function(x, y), self.compute_gradients(x, y)

    def back_propagation(self, x, y):
        m = x.shape[0]
        delta3 = self.a3-y
        # delta2 = delta3*Theta2.*sigmoidGradient([1 z2])
        delta2 = np.multiply(np.dot(delta3, self.Theta2), NeuralNetwork.sigmoid_gradient(np.hstack((np.ones(m).reshape(m, 1), self.z2))))

        # Exclude the first column
        delta2 = delta2[:, 1:]
        # Return delta3'*a2/m, delta2'*x/m
        self.Theta2_Grad = np.dot(delta3.transpose(), self.a2)/m
        self.Theta1_Grad = np.dot(delta2.transpose(), x)/m

    def get_params(self):
        # Get Weights

        params = np.concatenate((self.Theta1.ravel(), self.Theta2.ravel()))
        return params

    def set_params(self, params):
        # Set Weights

        theta1_start = 0
        theta1_end = self.hidden_layer_size * (self.input_layer_size+1)
        self.Theta1 = np.reshape(params[theta1_start:theta1_end], (self.hidden_layer_size, self.input_layer_size+1))
        theta2_end = theta1_end + (self.hidden_layer_size+1) * self.output_layer_size
        self.Theta2 = np.reshape(params[theta1_end:theta2_end], (self.output_layer_size, self.hidden_layer_size+1))

    def compute_gradients(self, x, y):
        self.back_propagation(x, y)
        return np.asarray(np.hstack((self.Theta1_Grad.ravel(), self.Theta2_Grad.ravel()))).flatten()

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_gradient(z):
        return np.multiply(NeuralNetwork.sigmoid(z), (1-NeuralNetwork.sigmoid(z)))

    @staticmethod
    def random_initialize(layer1, layer2):
        epsilon = 0.089
        return np.random.rand(layer1, layer2 + 1) * 2 * epsilon - epsilon


import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the sigmoid neuron
class SigmoidNeuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias with small random values
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()

    def forward(self, inputs):
        # Calculate the weighted sum of inputs and apply the sigmoid activation function
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return sigmoid(weighted_sum)

# Example usage
if __name__ == "__main__":
    # Create a sigmoid neuron with 3 input features
    sigmoid_neuron = SigmoidNeuron(3)

    # Input features
    inputs = np.array([0.5, 0.2, 0.8])

    # Get the output of the sigmoid neuron
    output = sigmoid_neuron.forward(inputs)

    print("Output of the sigmoid neuron:", output)

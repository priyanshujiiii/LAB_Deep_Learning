import numpy as np

class Perceptron:
    def __init__(self, num_inputs):
        # Initialize weights and bias with small random values
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    def activate(self, weighted_sum):
        # Define the step function as the activation function
        if weighted_sum >= 0:
            return 1
        else:
            return 0

    def forward(self, inputs):
        # Calculate the weighted sum of inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply the activation function
        return self.activate(weighted_sum)

# Example usage
if __name__ == "__main__":
    # Create a perceptron with 3 input features
    perceptron = Perceptron(3)

    # Input features
    inputs = np.array([0.5, 0.2, 0.8])

    # Get the output of the perceptron
    output = perceptron.forward(inputs)

    print("Output of the perceptron:", output)

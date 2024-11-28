import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# MLP class
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=100):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1)
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2)
        output = sigmoid(self.Z2)
        return output

    def backward(self, X, y, output):
        # Output layer error
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Hidden layer error
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.A1)

        # Update weights
        self.W2 += self.A1.T.dot(output_delta) * self.learning_rate
        self.W1 += X.T.dot(hidden_delta) * self.learning_rate

    def train(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 10 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")

# Function to create input table for a logical gate
def get_input_table(gate):
    if gate.lower() == 'and':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
    elif gate.lower() == 'or':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
    elif gate.lower() == 'xor':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
    else:
        raise ValueError("Unknown gate. Choose between 'AND', 'OR', 'XOR'.")
    return X, y

# Main function to run MLP
def run_mlp():
    # User inputs
    gate = input("Enter the logical gate (AND/OR/XOR): ")
    learning_rate = float(input("Enter learning rate: "))
    epochs = int(input("Enter number of epochs: "))

    # Create input table
    X, y = get_input_table(gate)

    # Initialize MLP
    mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=learning_rate, epochs=epochs)

    # Train MLP
    mlp.train(X, y)

    # Test the network
    while True:
        test_input = input("\nEnter test inputs (space-separated, e.g., '0 1') or 'q' to quit: ")
        if test_input.lower() == 'q':
            break
        test_input = np.array([int(x) for x in test_input.split()]).reshape(1, -1)
        prediction = mlp.forward(test_input)
        print(f"Prediction for input {test_input.flatten()}: {np.round(prediction)}")

# Run the MLP algorithm
if __name__ == "__main__":
    run_mlp()
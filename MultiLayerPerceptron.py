import numpy as np

# sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 1. input and output data for XOR
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# XOR target outputs
outputs = np.array([[0], [1], [1], [0]])

# 2. initialize weights and biases
np.random.seed(0)
input_size = 2
hidden_size = 4
output_size = 1

# weights from input to hidden layer
w_input_hidden = np.random.rand(input_size, hidden_size)
# Bias for hidden layer
b_hidden = np.random.rand(1, hidden_size)

# weights from hidden to output layer
w_hidden_output = np.random.rand(hidden_size, output_size)
# bias for output layer
b_output = np.random.rand(1, output_size)

# 3. training parameters
learning_rate = 0.1
epochs = 10000

# 4. training loop
for epoch in range(epochs):
    # ---- Forward pass ----
    # Input to hidden layer
    hidden_input = np.dot(inputs, w_input_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)

    # Hidden to output layer
    final_input = np.dot(hidden_output, w_hidden_output) + b_output
    predicted_output = sigmoid(final_input)

    # ---- Backward pass ----
    # Output layer error and gradient
    error = outputs - predicted_output
    d_output = error * sigmoid_derivative(predicted_output)

    # Hidden layer error and gradient
    error_hidden = d_output.dot(w_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # ---- Update weights and biases ----
    w_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    b_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    w_input_hidden += inputs.T.dot(d_hidden) * learning_rate
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.5f}")

# 5. test the trained network
print("\nFinal predictions (XOR):")
for x in inputs:
    hidden = sigmoid(np.dot(x, w_input_hidden) + b_hidden)
    output = sigmoid(np.dot(hidden, w_hidden_output) + b_output)
    print(f"{x} → {round(output[0][0], 3)}")

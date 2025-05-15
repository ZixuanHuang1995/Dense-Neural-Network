import numpy as np

# sigmoid 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 1. input data（AND Operation）
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# 2. Output label (result of AND)
outputs = np.array([[0], [0], [0], [1]])

# 3. Weight initialization (2 inputs to 1 output)
np.random.seed(0)
weights = np.random.rand(2, 1)
bias = np.random.rand(1)

# 4. Training the model
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):

    z = np.dot(inputs, weights) + bias
    predicted = sigmoid(z)

    error = outputs - predicted

    gradient = error * sigmoid_derivative(predicted)

    weights += np.dot(inputs.T, gradient) * learning_rate
    bias += np.sum(gradient) * learning_rate

    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 5. dispay final results
print("\nFinal prediction results: ")
for x in inputs:
    result = sigmoid(np.dot(x, weights) + bias)
    print(f"{x} → {round(result[0], 3)}")

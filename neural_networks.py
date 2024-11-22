import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate

        # Activation function and derivative
        if activation == 'tanh':
            self.activation = lambda x: np.tanh(x)
            self.activation_derivative = lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: self.activation(x) * (1 - self.activation(x))
        else:
            raise ValueError("Unsupported activation function")

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # For visualization
        self.hidden_activations = None
        self.gradients = None

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))  # Sigmoid for binary classification
        self.hidden_activations = self.A1  # Store hidden activations for visualization
        return self.A2

    def backward(self, X, y):
        # Compute gradients using the chain rule
        m = X.shape[0]  # Number of samples
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Store gradients for visualization
        self.gradients = {"W1": np.linalg.norm(dW1), "W2": np.linalg.norm(dW2)}

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int).reshape(-1, 1)  # Circular boundary
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Plot hidden features
    hidden_features = mlp.hidden_activations
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Layer Activations")

    # Plot input space decision boundary
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap='bwr', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Input Space Decision Boundary")

    # Plot gradients
    gradient_magnitude = mlp.gradients["W1"]
    ax_gradient.bar(["W1", "W2"], [gradient_magnitude, mlp.gradients["W2"]])
    ax_gradient.set_title("Gradient Magnitudes")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)

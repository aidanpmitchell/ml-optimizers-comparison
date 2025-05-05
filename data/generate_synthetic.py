from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

def generate_data():
    # set random_state for reproducibility to accurately compare optimization methods
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    return X, y

if __name__ == "__main__":
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.title("Synthetic 2D Classification Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    # ensure the model is in eval mode
    model.eval()

    # create a grid over the input space
    x_min, x_max = X[:, 0].min().item() - 0.5, X[:, 0].max().item() + 0.5
    y_min, y_max = X[:, 1].min().item() - 0.5, X[:, 1].max().item() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # flatten the grid and convert to torch tensor
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    # get model predictions
    with torch.no_grad():
        logits = model(grid_tensor)
        preds = torch.argmax(logits, dim=1).numpy()  # class predictions

    # reshape predictions to match the mesh
    Z = preds.reshape(xx.shape)

    # plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)

    # plot training data
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    plt.scatter(X[:, 0], X[:, 1], c=y_np, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()
import torch.nn as nn

# The model learns to classify data points in a 2D moon-shaped dataset
class SimpleMLP(nn.Module):
    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            # input: 2 features (from make_moons)
            nn.Linear(2, 16),

            # 1 hidden ReLU layer
            nn.ReLU(),

            # output: 2 classes (binary classification)
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

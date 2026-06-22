import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout=0.2
    ):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

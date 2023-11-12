from torch import nn

class SegNet(nn.Module):
    def __init__(self, d_in, d_out, d_hidden_list):
        super().__init__()
        layers = []

        layers.append(nn.Linear(d_in, d_hidden_list[0]))
        layers.append(nn.ReLU())


        for i in range(len(d_hidden_list) - 1):
            layers.append(nn.Linear(d_hidden_list[i], d_hidden_list[i + 1]))
            layers.append(nn.ReLU())

        # no activation in the last layer as we have a softmax lossa
        layers.append(nn.Linear(d_hidden_list[-1], d_out))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

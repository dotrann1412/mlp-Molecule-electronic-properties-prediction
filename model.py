from torch import nn 

class Regressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[], dropout=0.1):
        super(Regressor, self).__init__()

        self.fc = nn.Sequential()

        if len(hidden_layers) > 0:
            self.fc = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            for i in range(1, len(hidden_layers)):
                self.fc.add_module(f'hidden_{i}', nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.fc.add_module(f'relu_{i}', nn.ReLU())
                self.fc.add_module(f'dropout_{i}', nn.Dropout(dropout))

            self.fc.add_module('output', nn.Linear(hidden_layers[-1], output_size))

        else:
            self.fc.add_module('output', nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.fc(x)
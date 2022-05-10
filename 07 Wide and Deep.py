import numpy as np
from torch import nn
import torch
from torch import tensor

class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super().__init__()
        self.linear1 = nn.Linear(8, 6)  # One in and one out
        self.linear2 = nn.Linear(6, 4)  # One in and one out
        self.linear3 = nn.Linear(4, 1)  # One in and one out
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = nn.functional.relu(self.linear1(x))
        out2 = nn.functional.relu(self.linear2(out1))
        y_pred = nn.functional.relu(self.linear3(out2))
        return y_pred

    def train(self, times):
        data = np.loadtxt('./Data/diabetes.csv', delimiter=',', dtype = np.float32)
        x_data = torch.from_numpy(data[:, 0:-1])
        y_data = torch.from_numpy(data[:, [-1]])

        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Training loop
        for epoch in range(times):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x_data)

            # Compute and print loss
            loss = criterion(y_pred, y_data)
            print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def predict(self, x):
        return self.forward(x).data[0][0]

# our model
model = Model()
model.train(1000)
print(model.predict(tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])))





# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor, nn
import numpy as np

class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

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

    def train(self, epochs):
        dataset = DiabetesDataset()
        train_loader = DataLoader(dataset=dataset,
                                batch_size=32,
                                shuffle=True,
                                num_workers=2)

        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(inputs)

                # Compute and print loss
                loss = criterion(y_pred, labels)
                print(f'Epoch {epoch + 1} | Batch: {i+1} | Loss: {loss.item():.4f}')

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def predict(self, x):
        return self.forward(x).data[0][0]

if __name__ == '__main__':  
    # our model
    model = Model()
    model.train(100)
    print(model.predict(tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])))
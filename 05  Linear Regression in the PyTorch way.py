from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# reduction的意思是维度要不要缩减，以及怎么缩减，有三个选项：

# 'none': no reduction will be applied.
# 'mean': the sum of the output will be divided by the number of elements in the output.
# 'sum': the output will be summed.
# 如果不设置reduction参数，默认是'mean'。

# loss_fn1 = torch.nn.MSELoss(reduction='none')
# loss1 = loss_fn1(a.float(), b.float())
# print(loss1)   # 输出结果：tensor([[ 4.,  9.],
#                #                 [25.,  4.]])
 
# loss_fn2 = torch.nn.MSELoss(reduction='sum')
# loss2 = loss_fn2(a.float(), b.float())
# print(loss2)   # 输出结果：tensor(42.)
 
 
# loss_fn3 = torch.nn.MSELoss(reduction='mean')
# loss3 = loss_fn3(a.float(), b.float())
# print(loss3)   # 输出结果：tensor(10.5000)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, y_pred.item())
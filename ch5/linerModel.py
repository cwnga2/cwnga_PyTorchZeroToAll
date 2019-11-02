import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class Model(torch.nn.Module):

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




#2. use loss functon and optimizer(for update w)
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    # x_data is array 一次放全部的data， 不像之前的 又多寫一個for loop 輸入
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    #print(epoch, loss.data[0])


    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() #optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
    loss.backward()
    optimizer.step()


# After training
hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = model(hour_var)
print("predict (after training)",  4, model(hour_var).data[0][0])

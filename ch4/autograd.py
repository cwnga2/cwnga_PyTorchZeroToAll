import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = Variable(torch.Tensor([0.0]), requires_grad = True)

#model
def forward(x):
    return x * w

def loss(x, y):
    pred_y = forward(x)
    return (pred_y - y) * (pred_y - y)

# manually cal gradient
# def gradient(x, y):
#    return 2 * x * (x * w - y)

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("w.data:", w.data, "w.grad.data:", w.grad.data)
        w.data = w.data - 0.01 * w.grad.data #before w = w - 0.01 *gradient(x_val, y_val)
        w.grad.data.zero_() #set 0

print("predict", 4, forward(4).data)

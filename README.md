# ch1

setup
select your pip version and os
https://pytorch.org/
```
pip3 install torch torchvision

python
>>> import torch
>>> print(torch.__version__)
1.3.0
#Ctrl+D to exit

```

# PyTorch Lecture 02: Linear Model
https://www.youtube.com/watch?v=l-Fe9Ekxxj4

```
hours,points
1,2
2,4
3,6
4,?

hy = x * w + b

loss function(MES) = (hy-y)^2 / #sample
if w ... function value
w = 0, loss = 18.7
w = 1, loss = 4.7
w = 2, loss = 0
w = 3, loss = 4.7
w = 4, loss = 18.7

so , find w make Min MES

in ch2/forward.py

1. define forward function(our model)
   y = x * w
2. define loss function
   (predict y - y) ^ 2

3. data set

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

x = 1, y = 2
x = 2, y = 4
x = 3, y = 6


4. random w (from 0 to 4, each step 0.1)
w = 0, loss = 18.7
w = 0.1, loss = ...
w = 1, loss = 4.7
w = 1.1, loss = ...
...

finally, get to w and loss function relation(cost graph)

```
![image](./ch2/loss.png)


# ch3 PyTorch Lecture 03: Gradient Descent
https://www.youtube.com/watch?v=b4Vyma9wPHo

Gradient(斜率)

goal: 
```
loss(w)  = 1/N [1...N] (hy(n) - y(n)) ^ 2
argmin loss(w)
  w
---------------
\     /
 \   /   #d_loss/d_w
  \_/

---------------
traing 
w = w - a * gradient(x, y)

a: learning rate  

gradient = d_loss/d_w
         = d/dw[( hy - y) ^2]
         = d/dw[( x*w - y) ^2]
         = d/dw[ (x^2)(w^2) - 2xwy - 2y]
         = 2w(x^2) - 2xy
         = 2x(x*w - y) 

w = w - a * 2x(x * w - y)

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val) #帶入不同的x, y 和 目前的w 計算斜率，  
        w = w - 0.01 * grad  # 目前的w - 斜率， 學習新的w
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)

    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

```
上面是透過不斷 
當斜率> 0， 就w 變小
當斜率< 0， 就w 變大

# ch4 PyTorch Lecture 04: Back-propagation and Autograd

Back propagation 目的是for 計算w 對loss function  斜率


# ch5 linear model with PyTorch

1. create model, model inherit torch.nn.Module
```
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out
    def forward(self, x):
        y_pred = self.linear(x) # NOTE: before use x * w
        return y_pred
```

2. create loss functon and optimizer(for update w)
```
criterion = torch.nn.MSELoss(size_average=False) # loss = (pred_y - y) * (pred_y - y
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #  w = w - grad * 0.01

```

3. forward, backward, step(update w)
```
    loss = criterion(y_pred, y_data)
    loss.backward()
    optimizer.step()
```


# ch6 PyTorch Lecture 06: Logistic Regression

before: predict score

now: predict pass/failure


```
linear -> sigmoid -> 0~1 
( >= 0.5: predict y = 1, else: predict y = 0 )

loss function change: 
linear 
   loss function(MES) = (hy-y)^2 / #sample
logistic 
   lost function (cross entropy loss) =  ylog(hy) + (1-y)log(1-hy)
```

code change 
```
#import functional for Sigmoid
import torch.nn.functional as F

# y change to be 0 or 1
#linear
#x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
#y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

#logistic
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[0.0], [0.0], [1.0]]))


#forword
#liner y_pred = self.linear(x)
y_pred = F.sigmoid(self.linear(x))

# criterion MES -> BCELoss
#linear criterion = torch.nn.MSELoss(size_average=False)
#logistic
criterion = torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


```

# ch7 wide and deep

wide: more parameters x
deep: more layer

- import from csv
```

import numpy as np
xy = np.loadtxt('../data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

```

- more layer
```

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        out1 = self.sigmoid(self.linear1(x))
        out2 = self.sigmoid(self.linear2(out1))
        out3 = self.sigmoid(self.linear3(out2))
        return out3 #predict

```

# ch8 data loader
- import Dataset, DataLoader
```
from torch.utils.data import Dataset, DataLoader
```
- customer data set

-- over write __getitem__, __len__
```
class CustomerDataset(Dataset):
    def __init__(self):
    def __getitem__(self, index):
        return # x data , y data

    def __len__(self):
        return (number)

dataset = CustomerDataset()
train_loader = DataLoader(
        dataset=dataset,
        batch_size = 32,
        shuffle=True
        )
```

-- enumerate
```
for i, data in enumerate(train_loader, 0):
    x_data, y_data = data
    x_data, y_data = Variable(x_data), Variable(y_data)
```


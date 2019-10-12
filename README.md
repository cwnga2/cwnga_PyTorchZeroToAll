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

w = w - a * 2x(x*w - y)



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






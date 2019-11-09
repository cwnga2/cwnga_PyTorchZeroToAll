import torch
from torch.autograd import Variable

loss = torch.nn.CrossEntropyLoss()

Y = Variable(torch.LongTensor([0]), requires_grad = False)
Y_pred1 = Variable(torch.Tensor([[2.0,1,0.1]]))


print(loss(Y_pred1, Y).data)


#0 times: 2 is ans
#1 times: 0 is ans
#2 timme: 1 is ans

Y = Variable(torch.LongTensor([2,0,1]), requires_grad = False)
Y_pred1 = Variable(torch.Tensor([
    [0.1, 0.2, 0.9], #max: 0.9, index 2 is ans
    [1.1, 0.1, 0.2], #max: 1.1, index 0 is ans
    [0.2, 2.1, 0.1], #max: 2.1, index 1 is ans
    ]))

print(loss(Y_pred1, Y).data)


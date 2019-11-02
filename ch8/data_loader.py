import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomerDataset(Dataset):
    def __init__(self):
        super(CustomerDataset, self).__init__()
        xy = np.loadtxt(
                '../data/diabetes.csv.gz',
                delimiter=',',
                dtype=np.float32
                )
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        #over write __getitem__
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        #over write __len__
        return self.len

dataset = CustomerDataset()
train_loader = DataLoader(
        dataset=dataset,
        batch_size = 32,
        shuffle=True,
        )


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

model = Model()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        x_data, y_data = Variable(inputs), Variable(labels)
        y_pred = model.forward(x_data)
        loss = criterion(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

newUserData  = Variable(torch.Tensor([
    -0.294118,0.487437,0.180328,-0.292929,0,0.00149028,-0.53117,-0.0333333 #first dtaa -> 0
    ]))
print(newUserData, "diabetes:",model(newUserData).data[0] > 0.5)

newUserData  = Variable(torch.Tensor([
    -0.882353,-0.0653266,0.147541,-0.373737,0,-0.0938897,-0.797609,-0.933333 #last data: 1
    ]))
print(newUserData, "diabetes:",model(newUserData).data[0] > 0.5)


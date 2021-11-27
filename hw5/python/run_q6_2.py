import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cpu')

max_iters = 50
# pick a batch size, learning rate
batch_size = 64
learning_rate = 0.01

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

train_x = np.array([np.reshape(x, (32, 32)) for x in train_x])
valid_x = np.array([np.reshape(x, (32, 32)) for x in valid_x])
test_x = np.array([np.reshape(x, (32, 32)) for x in test_x])

train_x_ts = torch.from_numpy(train_x).type(torch.float32).unsqueeze(1)
train_y_ts = torch.from_numpy(train_y).type(torch.LongTensor)
train_loader = DataLoader(TensorDataset(train_x_ts, train_y_ts), batch_size=batch_size, shuffle=True)

test_x_ts = torch.from_numpy(test_x).type(torch.float32).unsqueeze(1)
test_y_ts = torch.from_numpy(test_y).type(torch.LongTensor)
test_loader = DataLoader(TensorDataset(test_x_ts, test_y_ts), batch_size=batch_size, shuffle=False)

valid_x_ts = torch.from_numpy(valid_x).type(torch.float32).unsqueeze(1)
valid_y_ts = torch.from_numpy(valid_y).type(torch.LongTensor)
valid_loader = DataLoader(TensorDataset(valid_x_ts, valid_y_ts), batch_size=batch_size, shuffle=False)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(16*16*16, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 36))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 16*16*16)
        x = self.fc1(x)
        return x


model = NN()

train_loss = np.zeros(max_iters)
train_acc = np.zeros(max_iters)
valid_acc = np.zeros(max_iters)
valid_loss = np.zeros(max_iters)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    vloss = 0
    vacc = 0
    for data in train_loader:
        # get the inputs
        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        targets = torch.max(labels, 1)[1]

        # forward 
        y_pred = model(inputs)
        loss = nn.functional.cross_entropy(y_pred, targets)

        total_loss += loss.item()
        predicted = torch.max(y_pred, 1)[1]
        total_acc += predicted.eq(targets.data).cpu().sum().item()

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()
    
    for dataV in valid_loader:
        inputsV = torch.autograd.Variable(dataV[0])
        labelsV = torch.autograd.Variable(dataV[1])
        targetsV = torch.max(labelsV, 1)[1]
        # data = data.reshape(data.shape[0], -1)
        y_predV = model(inputsV)
        lossV = F.cross_entropy(y_predV, targetsV)
        vloss += loss.item()
        predictedV = torch.max(y_predV, 1)[1]
        vacc += predictedV.eq(targetsV.data).cpu().sum().item()

  
    train_acc[itr] = (total_acc/train_y.shape[0])*100
    train_loss[itr] = total_loss
    valid_loss[itr] = vloss
    valid_acc[itr] = (vacc/(valid_x.shape[0]))*100
    
    if itr % 2 == 0:
         print("Epoch: {:2d} \t Training: \t loss: {:.2f} \t acc : {:.2f}".format(itr, train_loss[itr], train_acc[itr]),"\t Validation: \t loss: {:.2f} \t acc : {:.2f}".format(valid_loss[itr], valid_acc[itr]))

n = range(max_iters)
plt.figure("Loss")
plt.plot(n, train_loss, label='training')
plt.plot(n, valid_loss, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

n = range(max_iters)
plt.figure("Accuracy")
plt.plot(n, train_acc, label='training')
plt.plot(n, valid_acc, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


print('Train accuracy: {}'.format(train_acc[-1]))
print('Validation accuracy: {}'.format(valid_acc[-1]))

torch.save(model.state_dict(), "q62_weights.pkl")
